// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.
#pragma once

#include <cstdint>



__device__ __forceinline__ bool filtered_selected_covers_common_labels(
    uint32_t target, uint32_t selected, uint32_t candidate, const uint32_t *offsets, const uint32_t *labels)
{
    uint32_t target_it = offsets[target];
    const uint32_t target_end = offsets[target + 1];
    uint32_t candidate_it = offsets[candidate];
    const uint32_t candidate_end = offsets[candidate + 1];
    const uint32_t selected_begin = offsets[selected];
    const uint32_t selected_end = offsets[selected + 1];
    while (target_it < target_end && candidate_it < candidate_end)
    {
        const uint32_t target_label = labels[target_it];
        const uint32_t candidate_label = labels[candidate_it];
        if (target_label < candidate_label)
            ++target_it;
        else if (candidate_label < target_label)
            ++candidate_it;
        else
        {
            bool found = false;
            for (uint32_t selected_it = selected_begin; selected_it < selected_end; ++selected_it)
            {
                if (labels[selected_it] == target_label)
                {
                    found = true;
                    break;
                }
                if (labels[selected_it] > target_label)
                    break;
            }
            if (!found)
                return false;
            ++target_it;
            ++candidate_it;
        }
    }
    return true;
}
