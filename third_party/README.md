# Third Party Dependencies

## DGGT

- Source: https://github.com/xiaomi-research/dggt
- Pinned commit: a3276d2bbe4cbb03bcc117830b1836110a27adeb
- Runtime integration: `drive_gaussian_world/geometry/dggt.py`

The upstream source and model retain their original LICENSE and NOTICE. Some files are derived from VGGT and retain the corresponding upstream terms. The frontend uses the native depth reconstruction path; tracking and diffusion refinement are not invoked by the cache adapter.

Initialize the Git submodule or use `scripts/setup_dggt_source.sh` when working from a source archive. Installation commands and pretrained checkpoint instructions are maintained in the project README.
