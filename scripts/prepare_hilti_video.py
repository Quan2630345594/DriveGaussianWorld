"""Extract one Hilti camera and rectify its Kannala-Brandt images to a video."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--topic', required=True)
    parser.add_argument('--camera', choices=[f'cam{i}' for i in range(5)], required=True)
    parser.add_argument('--start-seconds', type=float, default=10)
    parser.add_argument('--duration', type=float, default=20)
    parser.add_argument('--fps', type=float, default=10)
    parser.add_argument('--max-width', type=int, default=960)
    parser.add_argument('--rotate180', action='store_true', help='Make inverted camera images upright')
    parser.add_argument('--output', type=Path, default=Path('data/raw/hilti_parking_sample'))
    args = parser.parse_args()
    if (not np.isfinite([args.start_seconds, args.duration, args.fps]).all()
            or args.start_seconds < 0 or args.duration <= 0 or args.fps <= 0
            or args.max_width < 160):
        parser.error('Use start >= 0, duration/fps > 0, and max-width >= 160')
    video = args.output.with_suffix('.mp4')
    if video.exists() or (args.output.exists() and any(args.output.iterdir())):
        parser.error('Output already exists; choose a new output path')
    with args.calibration.open(encoding='utf-8') as stream:
        intrinsics = yaml.safe_load(stream)['sensors'][args.camera]['intrinsics']
    if intrinsics['type'] != 'KannalaBrandt':
        parser.error('This converter requires KannalaBrandt calibration')
    params = intrinsics['parameters']
    calibrated_width, calibrated_height = params['image_size']
    raw_dir = args.output / 'raw_frames'
    rectified_dir = args.output / 'frames'
    timestamps = []
    map_x = map_y = new_k = None
    encodings = set()
    source_size = None
    with AnyReader([args.bag]) as reader:
        connections = [c for c in reader.connections if c.topic == args.topic]
        if not connections or any(c.msgtype != 'sensor_msgs/msg/Image' for c in connections):
            parser.error('Select an existing sensor_msgs/Image topic')
        start = reader.start_time + round(args.start_seconds * 1e9)
        stop = min(reader.end_time, start + round(args.duration * 1e9))
        period = 1e9 / args.fps
        for connection, timestamp, data in reader.messages(connections, start, stop):
            # Anchor sampling to a fixed timeline instead of accumulating rounding drift.
            target = (timestamps[0]['bag_timestamp_ns'] + round(len(timestamps) * period)
                      if timestamps else start)
            if timestamps and timestamp < target - period / 2:
                continue
            message = reader.deserialize(data, connection.msgtype)
            encoding = message.encoding.lower()
            if encoding not in {'mono8', 'rgb8', 'bgr8', 'bayer_rggb8', 'bayer_bggr8',
                                'bayer_gbrg8', 'bayer_grbg8'}:
                raise ValueError(f'Unsupported image encoding: {encoding}')
            channels = 3 if encoding in {'rgb8', 'bgr8'} else 1
            pixels = np.frombuffer(message.data, dtype=np.uint8)
            pixels = pixels.reshape(message.height, message.step)
            image = pixels[:, :message.width * channels].reshape(
                message.height, message.width, channels)
            if channels == 1:
                image = image[:, :, 0]
            if encoding == 'rgb8':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif encoding.startswith('bayer_'):
                conversion = {'bayer_rggb8': cv2.COLOR_BayerRGGB2BGR,
                              'bayer_bggr8': cv2.COLOR_BayerBGGR2BGR,
                              'bayer_gbrg8': cv2.COLOR_BayerGBRG2BGR,
                              'bayer_grbg8': cv2.COLOR_BayerGRBG2BGR}
                image = cv2.cvtColor(image, conversion[encoding])
            size = (message.width, message.height)
            if source_size is not None and size != source_size:
                raise ValueError('Camera resolution changes within the selected sequence')
            if source_size is None:
                source_size = size
                # Hilti may record downsampled images; scale intrinsics with resolution.
                scale_x = message.width / calibrated_width
                scale_y = message.height / calibrated_height
                k = np.array([[params['fx'] * scale_x, 0, params['cx'] * scale_x],
                              [0, params['fy'] * scale_y, params['cy'] * scale_y],
                              [0, 0, 1]], dtype=float)
                distortion = np.array([params[f'k{i}'] for i in range(1, 5)])
                width = min(message.width, args.max_width) // 2 * 2
                height = round(message.height * width / message.width) // 2 * 2
                output_size = (width, height)
                new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    k, distortion, size, np.eye(3), balance=0, new_size=output_size)
                map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
                    k, distortion, np.eye(3), new_k, output_size, cv2.CV_32FC1)
                raw_dir.mkdir(parents=True)
                rectified_dir.mkdir()
            rectified = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
            if args.rotate180:
                rectified = cv2.rotate(rectified, cv2.ROTATE_180)
            name = f'{len(timestamps):06d}.png'
            if (not cv2.imwrite(str(raw_dir / name), image)
                    or not cv2.imwrite(str(rectified_dir / name), rectified)):
                raise OSError('Failed to save camera frames')
            encodings.add(encoding)
            timestamps.append(dict(file=name, bag_timestamp_ns=timestamp,
                                   camera_timestamp_ns=message.header.stamp.sec * 10**9
                                   + message.header.stamp.nanosec))
    if len(timestamps) < 2:
        raise ValueError('Selected interval contains fewer than two images')
    intervals = np.diff([row['bag_timestamp_ns'] for row in timestamps]) / 1e9
    if np.any(intervals > 1.5 / args.fps):
        raise ValueError('Selected camera has gaps or fps is too high; choose another interval/fps')
    if args.rotate180:
        new_k[0, 2] = output_size[0] - 1 - new_k[0, 2]
        new_k[1, 2] = output_size[1] - 1 - new_k[1, 2]
    report = dict(bag=str(args.bag.resolve()), topic=args.topic, camera=args.camera,
                  calibration=str(args.calibration.resolve()), start_seconds=args.start_seconds,
                  requested_duration=args.duration, fps=args.fps, count=len(timestamps),
                  encodings=sorted(encodings), source_size=source_size, output_size=output_size,
                  pinhole_intrinsics=new_k.tolist(), distortion_model='KannalaBrandt',
                  rotate180=args.rotate180,
                  rectified_camera_from_raw_camera=(np.diag([-1, -1, 1])
                                                   if args.rotate180 else np.eye(3)).tolist(),
                  interval_range_seconds=[float(intervals.min()), float(intervals.max())],
                  frames=timestamps, video=str(video.resolve()),
                  note='Fixed-rate timestamp sampling; original image and bag timestamps are retained.')
    subprocess.run(['ffmpeg', '-nostdin', '-n', '-loglevel', 'error', '-framerate',
                    str(args.fps), '-i', str(rectified_dir / '%06d.png'), '-c:v', 'libx264',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags',
                    '+faststart', str(video)], check=True)
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'frames'}, indent=2))


if __name__ == '__main__':
    main()
