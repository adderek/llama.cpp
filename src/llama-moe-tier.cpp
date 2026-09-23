#include "llama-moe-tier.h"

#include "llama-impl.h"

#include <cstring>
#include <thread>
#include <cstdlib>
#include <algorithm>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#endif

static const size_t LLAMA_MOE_TIER_ALIGN = 4096;

// experts are read one slab at a time; a single stream leaves most of the NVMe unused
static size_t llama_moe_tier_readers() {
    static size_t n = 0;
    if (n == 0) {
        const char * env = getenv("LLAMA_MOE_READERS");
        n = env ? std::max(1, atoi(env)) : 8;
    }
    return n;
}
#define LLAMA_MOE_TIER_READERS llama_moe_tier_readers()

llama_moe_tier_layer::~llama_moe_tier_layer() {
#if defined(__linux__)
    if (fd != -1) {
        close(fd);
    }
#endif
}

void llama_moe_tier_layer::init(int64_t n_warm_, int64_t n_slots_) {
    n_warm  = n_warm_;
    n_slots = n_slots_;

    slot_expert.assign(n_slots, -1);
}

void llama_moe_tier_layer::init_windows() {
    const int64_t n_frames = n_slots - n_warm;

    windows.clear();
    windows.push_back({ this, 0, (int32_t) std::min(n_slots, n_cold) });
    for (int64_t lo = n_slots; lo < n_cold; lo += n_frames) {
        windows.push_back({ this, (int32_t) lo, (int32_t) std::min(lo + n_frames, n_cold) });
    }
}

static bool llama_moe_tier_pread(int fd, uint8_t * dst, size_t offs, size_t len, uint8_t * bounce_aligned, size_t bounce_len) {
    while (len > 0) {
        const size_t want = std::min(len, bounce_len - 2*LLAMA_MOE_TIER_ALIGN);
        const size_t beg  = offs & ~(LLAMA_MOE_TIER_ALIGN - 1);
        const size_t span = ((offs - beg) + want + LLAMA_MOE_TIER_ALIGN - 1) & ~(LLAMA_MOE_TIER_ALIGN - 1);

        size_t done = 0;
        while (done < span) {
            const ssize_t r = pread(fd, bounce_aligned + done, span - done, beg + done);
            if (r <= 0) {
                if (r < 0 && errno == EINTR) {
                    continue;
                }
                LLAMA_LOG_ERROR("%s: read at %zu failed: %s\n", __func__, beg + done, strerror(errno));
                return false;
            }
            done += r;
        }

        memcpy(dst, bounce_aligned + (offs - beg), want);

        dst  += want;
        offs += want;
        len  -= want;
    }

    return true;
}

void llama_moe_tier_layer::finalize(int fd_) {
    fd = fd_;

    GGML_ASSERT(pending.size() == arenas.size());

    size_t slab_max = 0;
    for (size_t i = 0; i < arenas.size(); ++i) {
        arenas[i].data = (uint8_t *) pending[i]->data;
        GGML_ASSERT(arenas[i].data != nullptr);
        slab_max = std::max(slab_max, arenas[i].slab);
    }
    pending.clear();

    bounce.resize(LLAMA_MOE_TIER_READERS);
    bounce_aligned.resize(LLAMA_MOE_TIER_READERS);
    for (size_t i = 0; i < bounce.size(); ++i) {
        bounce[i].resize(slab_max + 2*LLAMA_MOE_TIER_ALIGN);
        bounce_aligned[i] = bounce[i].data() + (LLAMA_MOE_TIER_ALIGN - ((uintptr_t) bounce[i].data()) % LLAMA_MOE_TIER_ALIGN) % LLAMA_MOE_TIER_ALIGN;
    }

#if defined(__linux__)
    // the arena starts out holding the warm experts, which are a prefix of the cold ones
    for (const auto & a : arenas) {
        llama_moe_tier_pread(fd, a.data, a.offs, a.slab*n_slots, bounce_aligned[0], bounce[0].size());
    }
#endif
}

int llama_moe_tier_open_direct(int fd) {
#if defined(__linux__)
    char link[64];
    snprintf(link, sizeof(link), "/proc/self/fd/%d", fd);

    char path[4096];
    const ssize_t n = readlink(link, path, sizeof(path) - 1);
    if (n <= 0) {
        return -1;
    }
    path[n] = '\0';

    return open(path, O_RDONLY | O_DIRECT);
#else
    GGML_UNUSED(fd);
    return -1;
#endif
}

int32_t llama_moe_tier_layer::slot_for(int32_t expert, std::vector<std::pair<int32_t, int32_t>> & load) {
    n_slot_total++;

    if (expert < n_warm) {
        n_hit++;
        return expert;
    }

    for (int64_t s = n_warm; s < n_slots; ++s) {
        if (slot_expert[s] == expert) {
            n_hit++;
            return (int32_t) s;
        }
    }

    const int64_t n_frames = n_slots - n_warm;
    const int64_t slot     = n_warm + next_frame;

    next_frame = (next_frame + 1) % n_frames;
    slot_expert[slot] = expert;
    n_miss++;

    load.emplace_back((int32_t) slot, expert);

    return (int32_t) slot;
}

void llama_moe_tier_layer::load_slots(const std::vector<std::pair<int32_t, int32_t>> & load) {
    if (load.empty()) {
        return;
    }

    // (frame, expert) x tensor, spread over the readers so the device sees more than one request
    const size_t n_reads = load.size()*arenas.size();
    const size_t n_threads = std::min<size_t>(LLAMA_MOE_TIER_READERS, n_reads);

    auto read_range = [&](size_t ith) {
        for (size_t i = ith; i < n_reads; i += n_threads) {
            const auto & l = load[i / arenas.size()];
            const auto & a = arenas[i % arenas.size()];
            llama_moe_tier_pread(fd, a.data + (size_t) l.first*a.slab, a.offs + (size_t) l.second*a.slab, a.slab,
                    bounce_aligned[ith], bounce[ith].size());
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(n_threads - 1);
    for (size_t i = 1; i < n_threads; ++i) {
        workers.emplace_back(read_range, i);
    }
    read_range(0);
    for (auto & w : workers) {
        w.join();
    }
}

void llama_moe_tier_map_ids(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    GGML_UNUSED(nth);

    auto * layer = (llama_moe_tier_layer *) userdata;

    GGML_ASSERT(a->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    std::vector<std::pair<int32_t, int32_t>> load;

    for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
        const int32_t * src_row = (const int32_t *) ((const char *) a->data   + i1*a->nb[1]);
        int32_t       * dst_row =       (int32_t *) (      (char *) dst->data + i1*dst->nb[1]);

        for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
            const int32_t id = src_row[i0];

            if (id < 0) {
                layer->n_slot_total++;
                layer->n_slot_hot++;
                dst_row[i0] = id;
                continue;
            }

            const int64_t n_cold = layer->n_cold > 0 ? layer->n_cold : layer->n_slots;
            layer->hist[std::min<int64_t>(7, 8*id/n_cold)]++;

            dst_row[i0] = layer->slot_for(id, load);
        }
    }

    layer->load_slots(load);
}

void llama_moe_tier_map_window(struct ggml_tensor * dst, const struct ggml_tensor * a, const struct ggml_tensor * b, int ith, int nth, void * userdata) {
    if (ith != 0) {
        return;
    }
    GGML_UNUSED(b);
    GGML_UNUSED(nth);

    const auto * w     = (const llama_moe_tier_layer::window *) userdata;
    auto       * layer = w->layer;

    GGML_ASSERT(a->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_are_same_shape(a, dst));

    const bool first = w->lo == 0;

    std::vector<std::pair<int32_t, int32_t>> load;

    for (int64_t i1 = 0; i1 < a->ne[1]; ++i1) {
        const int32_t * src_row = (const int32_t *) ((const char *) a->data   + i1*a->nb[1]);
        int32_t       * dst_row =       (int32_t *) (      (char *) dst->data + i1*dst->nb[1]);

        for (int64_t i0 = 0; i0 < a->ne[0]; ++i0) {
            const int32_t id = src_row[i0];

            if (id < w->lo || id >= w->hi) {
                if (first) {
                    layer->n_slot_total++;
                    layer->n_slot_hot += id < 0;
                    if (id >= 0) {
                        layer->hist[std::min<int64_t>(7, 8*id/layer->n_cold)]++;
                    }
                }
                dst_row[i0] = -1;
                continue;
            }

            if (first) {
                layer->n_slot_total++;
                layer->hist[std::min<int64_t>(7, 8*id/layer->n_cold)]++;
            }

            if (id < layer->n_warm) {
                layer->n_hit++;
                dst_row[i0] = id;
                continue;
            }

            // every id of the window has a frame of its own, so nothing the pass needs is replaced
            const int32_t slot = (int32_t) (layer->n_warm + (id - std::max<int64_t>(w->lo, layer->n_warm)));
            if (layer->slot_expert[slot] != id) {
                layer->slot_expert[slot] = id;
                layer->n_miss++;
                load.emplace_back(slot, id);
            } else {
                layer->n_hit++;
            }
            dst_row[i0] = slot;
        }
    }

    layer->load_slots(load);
}
