# Reference Experiments

These measurements were collected on 2026-10-19 with the engineering subset: KITTI raw drives 0005 / 0013 / 0009, 32 sampled times per drive. Each split contains 26 overlapping windows. The DGGT cache used input width 140 and two observed frames; world models used history 3, horizon 4 and hidden dimension 96, with 200 optimizer steps each. These historical runs used CPU execution. They are not results of the current default CUDA configuration or a full-scale benchmark.

## Logged motion conditions

| Method | RGB L1 | PSNR dB | Ego ADE m | Ego FDE m | Gaussian distance m |
|---|---:|---:|---:|---:|---:|
| dggt_world | 0.2302 | 10.0486 | 0.2507 | 0.3870 | 3.4907 |
| rgb_only | 0.2278 | 10.0706 | 0.2690 | 0.4221 | 12.4786 |
| no_actions | 0.2276 | 10.0935 | 0.2731 | 0.4661 | 3.3182 |
| stereo_world | 0.2284 | 10.0773 | 0.2179 | 0.3526 | 4.7381 |
| Persistence | 0.2267 | 10.0868 | 0.1819 | 0.2799 | 1.8789 |
| Action warp | 0.2267 | 10.0868 | 0.1819 | 0.2799 | 6.0799 |

The learned model does not consistently outperform persistence. Gaussian targets are frontend pseudo-geometry; distances across different target backends are not direct accuracy rankings. Effective-sampling masks in this DGGT cache are nearly always valid, so an IoU of 1 is a degenerate support statistic, not evidence of occupancy or occlusion understanding.

## Independent depth diagnostics

The held-out log contains three selected LiDAR times. DGGT sparse patch depth MAE is 4.900 m across 135 matched patches; stereo reference MAE is 2.332 m across 99 patches. The support differs and patch medians may mix occlusion layers. These are diagnostics, not a standard dense-depth benchmark.

## Current verification

The device refactor is covered by numerical, causality, checkpoint, API and device-routing regression tests. Default CUDA failure is explicit when unavailable. No new large-scale GPU training was launched as part of the documentation and device refactor. Reproduce a new experiment with the current README commands and retain its actual configuration and metadata.
