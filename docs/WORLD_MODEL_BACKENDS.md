# World model backends

DriveGaussianWorld supports three interchangeable dynamics backends. All consume historical RGB, DGGT Gaussian tokens, valid-slot masks and speed, plus separately supplied future motion conditions. Outputs are RGB `[B,F,3,32,96]`, Gaussian tokens `[B,F,64,16]`, mask logits `[B,F,64]` and local ego increments `[B,F,3]`.

These implementations are research frameworks. No trained RSSM/diffusion quality or GPU resource comparison is claimed. The existing `World_Model.docx` describes the original GRU baseline and shared heads; this document describes the new backends.

## Shared interface

```python
from drive_gaussian_world.models import build_world_model

model = build_world_model("rssm", hidden=96, model_options={"stochastic": 32})
state = model.observe(observation)           # historical observations only
prediction = model.rollout(state, actions, dt)
loss, metrics = model.training_loss(observation, actions, dt, target)
```

`forward(observation, actions, dt)` is equivalent to observe + rollout. Future targets are accepted only by `training_loss`; neither observe nor rollout takes future observations. Encoders, output heads, the analytic ego-motion warp and reconstruction loss definitions are shared in code, with independently learned weights per model. All use the existing driving dataset contract and scene splits. Old checkpoints without a backend field load as GRU.

## Deterministic GRU (`gru`)

The history GRU encodes image features (64), masked Gaussian features (48) and normalized speed (1). Future transitions use `[a/5, yaw_rate, dt, speed/20]`. Predictions are deterministic. The original parameter names are preserved for checkpoint compatibility.

`L_world = L_RGB + 2 L_geometry + 0.2 L_attributes + 0.05 L_mask + 0.1 L_ego`.

## Continuous RSSM (`rssm`)

This backend implements a Dreamer-style recurrent state-space model with continuous diagonal Gaussian latents. It is not a full Dreamer reinforcement learning agent and does not implement DreamerV2/V3 discrete latents, actor, critic, reward or continuation heads.

- Deterministic transition: `h_next = GRU([z, action_condition], h)`.
- Prior: `p(z_next | h_next)` predicts a mean and `softplus(scale) + 0.1`.
- Training posterior: `q(z_next | h_next, encoded_next_observation)`.
- Reparameterized sampling: `z = mean + std * epsilon`.
- Shared heads decode an MLP projection of `[h_next, z_next]`.

Historical filtering also recurs through deterministic and stochastic state. Historical actions and time intervals are absent from the current observation contract; those inputs are zeroed during filtering, while historical speed remains observed. Future recurrence uses the supplied conditions. Adding historical controls/dt to the dataset contract would remove this approximation.

Training uses posterior reconstructions plus balanced KL:

```text
L_RSSM = L_world(posterior_rollout) + kl_weight * KL_balanced
KL_balanced = balance * KL(stopgrad(q) || p)
            + (1 - balance) * KL(q || stopgrad(p))
```

Each KL sums latent dimensions, is clamped by `free_nats` per batch element, then averaged across batch and time. Defaults: latent 32, weight 0.1, balance 0.8, free nats 1. Inference samples only the prior after history filtering. Validation always uses open-loop prior predictions, not posterior reconstructions.

## Conditional latent diffusion (`diffusion`)

This backend diffuses a learned compact scene latent, not full-resolution pixels and not DGGT's upstream image refinement. A scene encoder maps the same 113 features to a `hidden`-dimensional LayerNorm latent. A causal GRU summarizes historical context and future controls. A conditioner combines that context with the previous scene latent.

Training uses a cosine noise schedule and epsilon prediction:

```text
x_k = sqrt(alpha_bar_k) * stopgrad(z_target)
    + sqrt(1 - alpha_bar_k) * epsilon
L_noise = MSE(epsilon_theta(x_k, k, context), epsilon)
z_estimate = (x_k - sqrt(1-alpha_bar_k)*epsilon_theta) / sqrt(alpha_bar_k)
L_diffusion = L_world(decode(z_estimate))
            + reconstruction_weight * L_world(decode(z_target))
            + denoising_weight * L_noise
```

The target latent encoder learns through its reconstruction term, while denoising targets are detached. Target latent and previous true scene are teacher forced only during training; inference recursively uses sampled latents and predicted scene outputs. The reconstruction branch is an initial jointly learned codec, not a pretrained or separately benchmarked autoencoder. Latent collapse, codec quality and teacher-forcing exposure bias must be checked when training on more data.

Inference uses a subsampled DDIM update with eta=0, starting from Gaussian noise. Different initial noise produces different futures. Defaults: 100 noise levels and 20 denoiser evaluations per future step. The final noise level has low signal due to the cosine schedule. Estimated clean latents are clamped to `[-5,5]` for numerical stability. Diffusion arithmetic and KL calculations use float32 where needed; GPU mixed precision remains configurable.

For stochastic backends, `sample=False` uses posterior/prior means (RSSM) or zero initial diffusion noise (diffusion). These are deterministic diagnostics, not estimates of the mean decoded future. `sample_predictions(..., count=K, seed=S)` returns `[K,B,F,...]` and supports reproducible sampling through a local generator. GRU remains deterministic without these arguments.

## Configuration and execution

Single-backend experiment configs are `configs/world_models/{gru,rssm,diffusion}.json`. The common training CLI accepts `--backend` and JSON `--model-options`; device and precision flags apply to every backend. Checkpoints save backend name and all backend options; evaluation, demo and API instantiate the correct model automatically.

```bash
python -m drive_gaussian_world train \
  --dataset data/processed/kitti_dggt --output runs/rssm \
  --backend rssm --model-options '{"stochastic":32,"kl_weight":0.1}' \
  --device cuda:0 --precision fp16

python -m drive_gaussian_world train \
  --dataset data/processed/kitti_dggt --output runs/diffusion \
  --backend diffusion --model-options '{"diffusion_steps":100,"sampling_steps":20}' \
  --device cuda:0 --precision fp16

# Starts real training: run only when data and compute have been allocated.
python -m drive_gaussian_world experiments \
  --dataset data/processed/kitti_dggt --config configs/comparison.json \
  --output runs/model_comparison

python -m drive_gaussian_world evaluate \
  --dataset data/processed/kitti_dggt --checkpoint runs/rssm/best.pt \
  --samples 5 --seed 42 --device cuda:0 --output runs/rssm/test_k5.json
```

Each checkpoint is selected by the same open-loop validation reconstruction objective. Validation RNG is fixed and restored so checkpoint selection has repeatable stochastic draws without changing training RNG. Training objectives differ by model and must not be ranked directly against each other. Evaluation records backend, K, seed and sampling options; with K>1 it adds mean-sample ADE, best-of-K ADE and trajectory spread. The ordinary metric table continues to report the first sample rather than an oracle selection. K samples independently repeat history filtering as well as future sampling. Diversity is not a quality or safety score.

## Future comparison protocol

Use identical DGGT caches, log splits, history/horizon, batch size and seed sets. The suite uses identical hidden sizes and optimizer step counts, which do **not** imply equal parameter counts or compute. Training summaries record parameter counts and wall time. Additional matched-parameter and matched-compute experiments are needed before claims about model families.

After resource allocation, measure synchronized latency, peak CUDA allocated/reserved memory, samples/second and training wall time at matched input sizes; sweep diffusion sampling steps and use the same K for stochastic quality metrics. Run multiple seeds, longer horizons, codec/latent diagnostics and action perturbations. Add independent geometry ground truth and broader scene-disjoint data. The current motion conditions are OXTS-derived proxies, not issued vehicle controls. No actor learning or closed-loop driving performance follows from these frameworks.

## Deferred GPU profiling

An opt-in profiler is available for later resource allocation. It is not invoked by training, tests or dataset collection. Run each backend in a fresh process, with identical batch/history/horizon/precision, then sweep diffusion sampling steps through `--model-options`:

```bash
python -m drive_gaussian_world profile --backend gru --device cuda:0 --output runs/profile_gru.json
python -m drive_gaussian_world profile --backend rssm --device cuda:0 --output runs/profile_rssm.json
python -m drive_gaussian_world profile --backend diffusion --device cuda:0 \
  --model-options '{"diffusion_steps":100,"sampling_steps":20}' --output runs/profile_diffusion.json
```

Reports include parameter count, synchronized mean/median inference latency, throughput and peak CUDA allocated/reserved bytes. Inputs are synthetic cached scenes; measurements include historical encoding and future sampling, and exclude DGGT, data loading and training. CUDA allocator statistics do not represent total board memory. No GPU profiling result is provided by this framework delivery.

## Framework validation

On 2026-10-21, the server's Python 3.10 / PyTorch 2.4.1+cu124 environment passed 53 tests. These cover causal inference boundaries, backend gradients, seeded sampling, action response, old/new checkpoints, one-step fixture training, evaluation/demo/API integration, deferred profiler contracts, dataset byte budgets, CRC/SHA recording and interrupted download recovery. Tests use small tensors without GPU execution. No large training, DGGT rerun or CUDA memory/latency benchmark was performed.

## References

- [PlaNet: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551): deterministic and stochastic recurrent state-space components.
- [DreamerV2: Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193): Dreamer family context; this implementation uses continuous rather than discrete latents.
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239): noise prediction training.
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502): subsampled deterministic reverse updates.
