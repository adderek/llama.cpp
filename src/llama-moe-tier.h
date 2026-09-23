#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// Demand paging for the cold experts of one MoE layer, with the model file as the
// backing store and the arena as the page frames.
//
// The first n_warm frames are pinned: they hold the experts the router uses most,
// which tools/moe-tier has sorted to the front. The frames after them are replaced
// round-robin, and an expert that is not resident is paged in with O_DIRECT, so it
// never enters the page cache: a fault on a full page cache costs 18.7 ms per 2 MiB,
// a direct read of the same 2 MiB costs 0.54 ms.
//
// This is not a victim cache: nothing is evicted from a level above, experts are
// fetched on demand. The graph only uses it when the ubatch routes to no more experts
// than there are replaceable frames, so a paged-in expert cannot be replaced while the
// call still needs it.
struct llama_moe_tier_layer {
    struct arena {
        uint8_t * data = nullptr; // arena base, holds n_slots experts
        size_t    slab = 0;       // bytes per expert
        size_t    offs = 0;       // file offset of cold expert 0
    };

    int     bid     = -1; // layer this state belongs to
    int     fd      = -1; // model file, opened with O_DIRECT
    int64_t n_warm  = 0;
    int64_t n_cold  = 0; // cold experts in the file
    int64_t n_slots = 0;

    std::vector<arena>   arenas;      // one per expert tensor of the layer
    std::vector<int32_t> slot_expert; // which expert each frame holds, -1 if none
    int64_t              next_frame = 0;

    std::vector<struct ggml_tensor *> pending; // arenas whose data pointer is known only after load

    // one bounce buffer per reader thread: O_DIRECT needs aligned reads and the tensor data is not
    std::vector<std::vector<uint8_t>> bounce;
    std::vector<uint8_t *>            bounce_aligned;

    int64_t n_miss = 0; // experts paged in from the file so far
    int64_t n_hit  = 0;
    int64_t n_slot_total = 0; // routed slots seen, including the ones served from VRAM
    int64_t n_slot_hot   = 0; // slots that belong to the hot half
    int64_t hist[8] = {0};    // cold ids by eighth of the cold range

    ~llama_moe_tier_layer();

    void init(int64_t n_warm, int64_t n_slots);

    // called once the arena buffer is allocated: bind the data pointers and read the
    // warm experts from the file
    void finalize(int fd);

    // frame holding this expert; a frame that still has to be paged in is added to `load`
    int32_t slot_for(int32_t expert, std::vector<std::pair<int32_t, int32_t>> & load);

    // page in the queued (frame, expert) pairs, several at a time
    void load_slots(const std::vector<std::pair<int32_t, int32_t>> & load);
};

// ggml custom op: rewrite cold expert ids into arena slots, loading what is missing.
// userdata is the llama_moe_tier_layer of this layer.
void llama_moe_tier_map_ids(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);

// open a second handle to the file behind fd, with O_DIRECT; -1 if that is not possible
int llama_moe_tier_open_direct(int fd);
