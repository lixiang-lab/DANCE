# Artifact scope

Immutable release: [`artifact-v1.0.12`](https://github.com/lixiang-lab/DANCE/releases/tag/artifact-v1.0.12).

| Paper capability | Released implementation | Entry point |
|---|---|---|
| Standard in-memory GPU Vamana | DANCE standard builder | `gpu_vamana_memory_index` |
| Standard sharded GPU Vamana | DiskANN partition and shard pipeline with DANCE shard construction | `build_disk_index` |
| GPU-FilteredVamana | Fixed complete production implementation | `build_memory_index --gpu_filtered` |
| Sharded GPU-FilteredVamana | Fixed complete production implementation in the DiskANN SSD pipeline | `build_disk_index --gpu_filtered` |
| GPU-StitchedVamana | Per-label GPU construction, global-ID remapping, stitching, sidecar generation, and conditional final prune | `gpu_stitched_vamana_index` |
| CPU memory search | Upstream DiskANN search | `search_memory_index` |
| SSD search | Upstream DiskANN search | `search_disk_index` |

The release contains neither competitor implementations nor paper-only experimental variants. GPU-FilteredVamana defaults to three full-graph synchronous rounds with 32-entry per-label search, eight retained per-label outputs, 64-entry union search, deterministic label-quota compaction, and exact CSR reverse repair. The retained active-frontier scheduler is opt-in and disabled by default. GPU-FilteredVamana provides shared-graph overlapping-label construction. GPU-StitchedVamana supports both partitioned and overlapping-label construction through label-local graphs and stitching. Standard and filtered release gates are reported in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The DANCE source files and the patch in `integrations/` are the authoritative artifact. Apply the patch to the supported upstream DiskANN commit as described in [INTEGRATION.md](INTEGRATION.md).
