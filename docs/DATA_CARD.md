# KITTI Driving Data Card

## Source and license

Data comes from the official [KITTI raw dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php) public S3 objects. KITTI is distributed under CC BY-NC-SA 3.0; retain the attribution and follow its non-commercial and share-alike terms. The default repository excludes raw images, point clouds, model weights and generated datasets.

## Original engineering subset

| Split | Drive | Sampled times | Mean forward speed |
|---|---|---:|---:|
| Train | `2011_09_26_drive_0005_sync` | 32 | 4.17 m/s |
| Validation | `2011_09_26_drive_0013_sync` | 32 | 11.42 m/s |
| Test | `2011_09_26_drive_0009_sync` | 32 | 10.85 m/s |

The example selects original frame indices 0, 4, …, 124 from each drive, approximately 12.8 seconds per log. It collects synchronized left/right RGB, OXTS, timestamps and calibration. Three LiDAR frames per drive are reserved for independent sparse-depth diagnostics; LiDAR is not used for world-model inputs or stereo scale anchoring.

## Expanded driving subset

`configs/data/kitti_driving.json` declares 20 complete moving logs: 16 train, 2 validation and 2 test. `download-drives` samples OXTS forward speed before accepting each log; the stationary candidate `0057` was excluded. Natural stops within driving logs remain valid road scenes. All logs are from 2011-09-26, so recording-date and geographic limitations still apply.

The expanded collector preserves all available color stereo frames, OXTS, timestamps, calibration and LiDAR. LiDAR remains available for independent evaluation and is not automatically used as a world-model input. It checks total ZIP sizes and a conservative peak consisting of all selected raw files plus the two largest temporary ZIPs and a metadata reserve, against a 49 GB decimal budget below the requested 50 GB ceiling. Temporary ZIPs are removed after selected members pass CRC and SHA256 recording.

Actual planned bytes, frame IDs, sampled speeds and archive provenance are saved in `data/kitti_driving/download_plan.json`. Download state is in `status.json`; only `COMPLETED` together with the final `collection.json` denotes a finished collection. `prepare --all-collected` uses these declared log splits and frames, rather than selecting unrelated files found in the directory. The original subset and original experiment results remain separate.

## Completed collection

The expanded collection completed and passed raw-source validation on 2026-10-21. It contains **20 driving logs**, **7,915 synchronized camera/OXTS times**, **15,830 RGB images** and **7,911 LiDAR scans**. Source ZIPs total 32.37 GB; retained sensors and metadata occupy approximately **28.02 GB** (decimal), below 50 GB. Splits contain 16 training, 2 validation and 2 test logs. Per-log source ZIP SHA256, full-log speed summaries and counts are in [DATASET_MANIFEST.json](DATASET_MANIFEST.json).

The official `0009` source ZIP contains no LiDAR scans for image IDs **177, 178, 179, 180**. The downloaded point-cloud list matches the source archive exactly; camera/OXTS frames remain available. Independent geometry evaluation must use the recorded LiDAR availability rather than inventing point clouds or treating the absence as a transfer failure.

Validation checks complete RGB/OXTS frame IDs, LiDAR source-member availability, increasing camera/OXTS timestamps and alignment within 50 ms, finite OXTS, whole-log mean motion above 1 m/s, PNG signatures/dimensions and XYZI record sizes. It does not run scene reconstruction or train a model. Full raw verification is recorded locally in `data/kitti_driving/raw_validation.json`; per-member CRC32/SHA256 and source archive hashes are in `collection.json`. The new dataset has not been used for a new DGGT cache or model performance claim.

## Collection and integrity

`collect` reads ZIP central directories and requests only selected public members. Each member is checked against its ZIP CRC32, with a locally computed SHA256 saved in `data/kitti/collection.json`. Hashes identify exact downloaded bytes; they are not upstream digital signatures. Collection checks OXTS motion before fetching the image interval. Preparation requires the complete configured frame set, valid time ordering and sensor alignment.

## Labels and geometry

Actions are `[acceleration, yaw_rate]` proxies derived from adjacent OXTS measurements, not issued throttle/brake/steering commands. Logged future actions may be supplied as conditions during conditional evaluation; the no-future-action protocol uses a constant-speed straight scenario.

DGGT geometry is reconstructed independently at each observed time and converted to the current optical camera frame. Stereo depth provides a historical median scale anchor. This is not full camera/IMU/vehicle registration, independent ground-truth geometry or occupancy. Gaussian supervision remains frontend pseudo-geometry. Dynamic probabilities are DGGT estimates rather than annotated actor trajectories.

## Split limitations

Splits use complete logs, never randomly shuffled frames. Logs share a recording date and sensor platform, and geographically nearby routes may overlap. Windows within a split overlap temporally. This subset validates the engineering workflow and does not establish cross-city, cross-weather or cross-platform generalization. Scaling data requires a new log manifest, suitable scene/geographic splits, and regenerated cache provenance.

## Citation

Geiger, Lenz, Stiller and Urtasun. *Vision meets Robotics: The KITTI Dataset*. International Journal of Robotics Research, 2013.
