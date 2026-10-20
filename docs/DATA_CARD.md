# KITTI Driving Data Card

## Source and license

Data comes from the official [KITTI raw dataset](https://www.cvlibs.net/datasets/kitti/raw_data.php) public S3 objects. KITTI is distributed under CC BY-NC-SA 3.0; retain the attribution and follow its non-commercial and share-alike terms. The default repository excludes raw images, point clouds, model weights and generated datasets.

## Engineering subset

| Split | Drive | Sampled times | Mean forward speed |
|---|---|---:|---:|
| Train | `2011_09_26_drive_0005_sync` | 32 | 4.17 m/s |
| Validation | `2011_09_26_drive_0013_sync` | 32 | 11.42 m/s |
| Test | `2011_09_26_drive_0009_sync` | 32 | 10.85 m/s |

The example selects original frame indices 0, 4, …, 124 from each drive, approximately 12.8 seconds per log. It collects synchronized left/right RGB, OXTS, timestamps and calibration. Three LiDAR frames per drive are reserved for independent sparse-depth diagnostics; LiDAR is not used for world-model inputs or stereo scale anchoring.

## Collection and integrity

`collect` reads ZIP central directories and requests only selected public members. Each member is checked against its ZIP CRC32, with a locally computed SHA256 saved in `data/kitti/collection.json`. Hashes identify exact downloaded bytes; they are not upstream digital signatures. Collection checks OXTS motion before fetching the image interval. Preparation requires the complete configured frame set, valid time ordering and sensor alignment.

## Labels and geometry

Actions are `[acceleration, yaw_rate]` proxies derived from adjacent OXTS measurements, not issued throttle/brake/steering commands. Logged future actions may be supplied as conditions during conditional evaluation; the no-future-action protocol uses a constant-speed straight scenario.

DGGT geometry is reconstructed independently at each observed time and converted to the current optical camera frame. Stereo depth provides a historical median scale anchor. This is not full camera/IMU/vehicle registration, independent ground-truth geometry or occupancy. Gaussian supervision remains frontend pseudo-geometry. Dynamic probabilities are DGGT estimates rather than annotated actor trajectories.

## Split limitations

Splits use complete logs, never randomly shuffled frames. Logs share a recording date and sensor platform, and geographically nearby routes may overlap. Windows within a split overlap temporally. This subset validates the engineering workflow and does not establish cross-city, cross-weather or cross-platform generalization. Scaling data requires a new log manifest, suitable scene/geographic splits, and regenerated cache provenance.

## Citation

Geiger, Lenz, Stiller and Urtasun. *Vision meets Robotics: The KITTI Dataset*. International Journal of Robotics Research, 2013.
