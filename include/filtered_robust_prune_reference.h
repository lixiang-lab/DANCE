// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <unordered_set>
#include <vector>

namespace diskann
{
namespace filtered_reference
{

struct Candidate
{
    uint32_t id;
    float distance_to_target;
};




inline std::vector<uint32_t>
robust_prune(const std::vector<uint32_t> &target_labels, std::vector<Candidate> candidates, uint32_t degree,
             float alpha, const std::function<const std::vector<uint32_t> &(uint32_t)> &labels_for,
             const std::function<float(uint32_t, uint32_t)> &pair_distance)
{
    std::sort(candidates.begin(), candidates.end(), [](const Candidate &left, const Candidate &right) {
        return left.distance_to_target < right.distance_to_target ||
               (left.distance_to_target == right.distance_to_target && left.id < right.id);
    });
    std::unordered_set<uint32_t> seen_ids;
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](const Candidate &candidate) {
                         return !seen_ids.insert(candidate.id).second;
                     }),
                     candidates.end());

    std::vector<uint8_t> removed(candidates.size(), 0);
    std::vector<uint32_t> result;
    result.reserve(std::min<size_t>(degree, candidates.size()));
    for (size_t i = 0; i < candidates.size() && result.size() < degree; ++i)
    {
        if (removed[i])
            continue;
        result.push_back(candidates[i].id);
        const auto &selected_labels = labels_for(candidates[i].id);
        for (size_t j = i + 1; j < candidates.size(); ++j)
        {
            if (removed[j])
                continue;
            const auto &candidate_labels = labels_for(candidates[j].id);
            bool covered = true;
            auto target_it = target_labels.begin();
            auto candidate_it = candidate_labels.begin();
            while (target_it != target_labels.end() && candidate_it != candidate_labels.end())
            {
                if (*target_it < *candidate_it)
                    ++target_it;
                else if (*candidate_it < *target_it)
                    ++candidate_it;
                else
                {
                    if (!std::binary_search(selected_labels.begin(), selected_labels.end(), *target_it))
                    {
                        covered = false;
                        break;
                    }
                    ++target_it;
                    ++candidate_it;
                }
            }
            if (covered && alpha * pair_distance(candidates[i].id, candidates[j].id) <=
                               candidates[j].distance_to_target)
                removed[j] = 1;
        }
    }
    return result;
}

}
}
