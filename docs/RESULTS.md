# RSSM Pilot Results

On 2026-11-24, a single-GPU RSSM pilot used the existing causal DGGT cache: KITTI drives 0005 / 0013 / 0009 for train / validation / test respectively. Each log contained 32 sampled frames and 21 overlapping windows with history 4 and horizon 8. This is a small engineering pilot, not the expanded 20-log dataset or a cross-dataset experiment.

The model used hidden dimension 256, stochastic dimension 64, batch size 8 and 1,054,495 trainable parameters. Validation selected the checkpoint at step 150. Training stopped at step 525 after 15 validation checks without sufficient improvement, taking 34.8 seconds. PyTorch peak allocated memory was 93,521,408 bytes; this excludes driver/context overhead.

| Method | RGB L1 | PSNR dB | Ego ADE m | Gaussian distance m |
|---|---:|---:|---:|---:|
| RSSM with logged motion | 0.2541 | 9.4015 | 0.6289 | 5.6876 |
| Last-frame persistence | 0.2359 | 9.8104 | 0.3288 | 1.9317 |

The learned model did not beat persistence. Its small training split and restricted decoder do not support conclusions about Dreamer architecture quality. Geometry metrics use frontend pseudo-labels. The persistence ego metric uses the analytical motion condition, not a frozen ego position. Mask IoU was 1.0 and is not an occupancy or occlusion metric.

Artifacts are retained outside Git under `output/rssm_pilot_20261124`: protocol, metrics, best checkpoint, test evaluation, learning curve and rollout GIF. This pilot was not repeated or tuned after viewing the held-out test result.
