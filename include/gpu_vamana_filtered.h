// Copyright (c) 2026 DANCE Authors. Licensed under the MIT License.
#ifndef GPU_VAMANA_FILTERED_H
#define GPU_VAMANA_FILTERED_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GPUFilteredVamanaStats
{
    double gpu_build_seconds;
    double label_h2d_seconds;
    double graph_build_seconds;
    double diagnostics_seconds;
    double d2h_seconds;
    double peak_gpu_mem_gb;
    double backbone_seconds;
    double task_generation_seconds;
    double filtered_search_seconds;
    double ordinary_search_seconds;
    double merge_seconds;
    double forward_prune_seconds;
    double csr_reverse_seconds;
    double reverse_prune_seconds;
    double compaction_seconds;

    uint64_t filtered_search_distance_count;
    uint64_t unfiltered_search_distance_count;
    uint64_t label_check_count;
    uint64_t label_reject_count;
    uint64_t filtered_candidate_count;
    uint64_t unfiltered_candidate_count;
    uint64_t merged_candidate_count;
    uint64_t active_rows;
    uint64_t original_neighbor_count;
    uint64_t filtered_seed_count;
    uint64_t filtered_visited_count;
    uint64_t filtered_visited_reserved_count;
    uint64_t filtered_top_reserved_count;
    uint64_t bridge_reserved_count;
    uint64_t unfiltered_reserved_count;
    uint64_t filtered_reserved_count;
    uint64_t label_occlusion_blocked_count;
    uint64_t geometry_occluded_count;
    uint64_t candidate_rejected_by_label_count;
    uint64_t refill_count;
    uint64_t filtered_prune_reject_count;
    uint64_t universal_label_pass_count;
    uint64_t final_prune_rows;
    uint64_t reverse_edges;
    uint64_t reverse_touched_rows;
    uint64_t filtered_reverse_edges;
    uint64_t filtered_reverse_touched_rows;
    uint64_t filtered_reverse_candidates_kept;
    uint64_t filtered_reverse_pruned_rows;

    double avg_common_label_out_degree;
    double common_label_degree_before_avg;
    double common_label_degree_after_avg;
    double active_rows_frac;
    uint32_t low_common_label_degree_count;
    uint32_t low_common_before;
    uint32_t low_common_after;
    uint32_t out_degree_min;
    uint32_t out_degree_max;
    double out_degree_avg;
    uint32_t common_label_degree_min;
    uint32_t common_label_degree_max;
    uint32_t invalid_neighbor_count;
    uint32_t self_loop_count;
    double prune_output_degree_avg;
    uint32_t filtered_seed_source;
    uint32_t semantic_stage;
    uint32_t semantic_complete;
} GPUFilteredVamanaStats;

int gpu_vamana_filtered_build_uint8(const uint8_t *h_data,
                                    uint32_t num_points,
                                    uint32_t dim,
                                    uint32_t R,
                                    uint32_t L,
                                    uint32_t filtered_L,
                                    uint32_t C,
                                    uint32_t STEPS,
                                    float alpha,
                                    const uint32_t *point_label_offsets,
                                    const uint32_t *point_labels,
                                    uint32_t total_label_count,
                                    const uint32_t *label_to_start_id,
                                    uint32_t num_labels,
                                    uint32_t universal_label_id,
                                    uint32_t global_start_id,
                                    const uint32_t *query_anchor_offsets,
                                    const uint32_t *query_anchor_ids,
                                    uint32_t query_anchor_count,
                                    uint32_t *h_graph,
                                    uint32_t *h_degree);

int gpu_vamana_filtered_build_float(const float *h_data,
                                    uint32_t num_points,
                                    uint32_t dim,
                                    uint32_t R,
                                    uint32_t L,
                                    uint32_t filtered_L,
                                    uint32_t C,
                                    uint32_t STEPS,
                                    float alpha,
                                    const uint32_t *point_label_offsets,
                                    const uint32_t *point_labels,
                                    uint32_t total_label_count,
                                    const uint32_t *label_to_start_id,
                                    uint32_t num_labels,
                                    uint32_t universal_label_id,
                                    uint32_t global_start_id,
                                    const uint32_t *query_anchor_offsets,
                                    const uint32_t *query_anchor_ids,
                                    uint32_t query_anchor_count,
                                    uint32_t *h_graph,
                                    uint32_t *h_degree);

void gpu_vamana_filtered_get_last_stats(GPUFilteredVamanaStats *stats);

#ifdef __cplusplus
}
#endif

#endif
