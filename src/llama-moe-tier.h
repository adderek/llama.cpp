#pragma once

#include "ggml.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// Cold experts of one MoE layer, kept in a RAM arena instead of an mmap.
//
// The arena holds the first n_warm experts, which are the most used ones after the
// model has been through tools/moe-tier, plus a few slots for the rest. An expert that
// is not in the arena is read straight from the file with O_DIRECT, so it never enters
// the page cache: a fault on a full page cache costs 18.7 ms per 2 MiB, a direct read
// of the same 2 MiB costs 0.54 ms.
//
// Used for decoding only, where a token routes to at most n_expert_used experts per
// layer and the victim slots cannot be needed twice in one call. Prompt processing
// touches every expert anyway and keeps using the mmap.
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
    std::vector<int32_t> slot_expert; // which expert each victim slot holds, -1 if none
    int64_t              next_victim = 0;

    std::vector<struct ggml_tensor *> pending; // arenas whose data pointer is known only after load

    // one bounce buffer per reader thread: O_DIRECT needs aligned reads and the tensor data is not
    std::vector<std::vector<uint8_t>> bounce;
    std::vector<uint8_t *>            bounce_aligned;

    int64_t n_miss = 0; // experts read from the file so far
    int64_t n_hit  = 0;
    int64_t n_slot_total = 0; // routed slots seen, including the ones served from VRAM
    int64_t n_slot_hot   = 0; // slots that belong to the hot half
    int64_t hist[8] = {0};    // cold ids by eighth of the cold range

    ~llama_moe_tier_layer();

    void init(int64_t n_warm, int64_t n_slots);

    // called once the arena buffer is allocated: bind the data pointers and read the
    // warm experts from the file
    void finalize(int fd);

    // arena slot for this expert; a slot that still has to be read is added to `load`
    int32_t slot_for(int32_t expert, std::vector<std::pair<int32_t, int32_t>> & load);

    // read the queued (slot, expert) pairs, several at a time
    void load_slots(const std::vector<std::pair<int32_t, int32_t>> & load);
};

// ggml custom op: rewrite cold expert ids into arena slots, loading what is missing.
// userdata is the llama_moe_tier_layer of this layer.
void llama_moe_tier_map_ids(struct ggml_tensor * dst, const struct ggml_tensor * a, int ith, int nth, void * userdata);

// open a second handle to the file behind fd, with O_DIRECT; -1 if that is not possible
int llama_moe_tier_open_direct(int fd);
