// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.



#pragma once

#include <cstdint>
#include <string>

namespace diskann
{

int merge_shards_gpu(const std::string &vamana_prefix, const std::string &vamana_suffix,
                     const std::string &idmaps_prefix, const std::string &idmaps_suffix,
                     uint64_t nshards, uint32_t max_degree, const std::string &output_vamana,
                     const std::string &medoids_file);

}
