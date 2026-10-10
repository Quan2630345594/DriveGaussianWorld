#!/usr/bin/env python3
"""Prepare recorded ParkingE2E observations, never independent ground truth.

Run with the parkings Python, LD_PRELOAD unset and PYTHONNOUSERSITE=1.
Upstream OCAM equations and extrinsic conventions are reproduced from the
pinned GPL-3.0 ParkingE2E sources saved alongside the calibration/license.
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import urllib.request

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation, Slerp

COMMIT = "acec907b9e09b052120f7861a48d261c093ab5bb"
BASE = f"https://raw.githubusercontent.com/qintonguav/ParkingE2E/{COMMIT}/"
CAMS = ("front", "back", "left", "right")
LOC = "/localization/estimation"
TOPICS = {c: f"/driver/fisheye/{c}/compressed" for c in CAMS}
SOURCE = "actual_recorded_reference_estimated_localization_not_independent_ground_truth"


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def stamp(message):
    return int(message.header.stamp.sec) * 10**9 + int(message.header.stamp.nanosec)


def xyz(v):
    return [v.x, v.y, v.z]


def open_bag(path):
    reader = AnyReader([path], default_typestore=get_typestore(Stores.ROS1_NOETIC))
    reader.open()
    # Register the bag's own definitions, including all cyber_msgs dependencies.
    definitions = {}
    for c in reader.connections:
        definitions.update(get_types_from_msg(c.msgdef.data, c.msgtype))
    reader.typestore.register(definitions)
    return reader


def download_calibration(root):
    if (root / "origin.json").exists():
        origin = json.loads((root / "origin.json").read_text())
        if origin["commit"] != COMMIT:
            raise RuntimeError("Existing calibration has a different upstream commit")
        for item in origin["files"]:
            if hashlib.sha256((root/item["path"]).read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError("Existing calibration failed its recorded SHA256")
        return origin
    paths = [f"fisheye_undistort/para/{c}/calib_results_{c}.txt" for c in CAMS]
    paths += ["utils/camera_config_right_hand.json", "utils/camera_utils.py",
              "utils/pose_utils.py", "fisheye_undistort/fish_eye_camera.py",
              "fisheye_undistort/fish_cam_process.py", "LICENSE"]
    records = []
    for name in paths:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(BASE + name, timeout=60).read()
        if target.exists():
            if target.read_bytes() != data:
                raise RuntimeError(f"Refusing to overwrite differing calibration: {target}")
        else:
            with target.open("xb") as f:
                f.write(data)
        records.append(dict(path=name, url=BASE + name, sha256=hashlib.sha256(data).hexdigest()))
    origin = dict(repository="https://github.com/qintonguav/ParkingE2E", commit=COMMIT,
                  license="GPL-3.0", files=records)
    if not (root / "origin.json").exists():
        write_json(root / "origin.json", origin)
    return origin


def inspect(bag, out):
    if not (out / "message_samples.json").exists():
        reader = open_bag(bag)
        samples = {}
        try:
            topics = [LOC, *TOPICS.values(), "/rock_can/speed_feedback"]
            for topic in topics:
                conn = next(c for c in reader.connections if c.topic == topic)
                _, bt, data = next(reader.messages(connections=[conn]))
                m = reader.deserialize(data,conn.msgtype)
                sample = dict(bag_time_ns=bt,header_time_ns=stamp(m),frame_id=m.header.frame_id,
                              definition=conn.msgdef.data)
                if topic == LOC:
                    sample.update(position=xyz(m.pose.position),velocity_linear=xyz(m.velocity.linear),status=int(m.status))
                if topic in TOPICS.values():
                    sample.update(format=m.format)
                samples[topic] = sample
        finally:
            reader.close()
        write_json(out / "message_samples.json",samples)
    if (out / "inspection.json").exists():
        logging.info("Reusing completed inspection; no reference files overwritten")
        return
    for name in ["reference_full.npz","image_index.npz","can_speed_full.npz"]:
        if (out/name).exists():
            raise RuntimeError(f"Refusing to overwrite incomplete/preexisting inspection artifact: {out/name}")
    rows, images, can, tf = [], {c: [] for c in CAMS}, [], []
    reader = open_bag(bag)
    try:
        metadata = dict(start_time_ns=reader.start_time, end_time_ns=reader.end_time,
                        duration_s=(reader.end_time-reader.start_time)/1e9,
                        bag=str(bag.resolve()), bag_bytes=bag.stat().st_size,
                        topics=[dict(topic=c.topic, msgtype=c.msgtype, count=c.msgcount)
                                for c in reader.connections])
        wanted = [c for c in reader.connections if c.topic in
                  [LOC, *TOPICS.values(), "/rock_can/speed_feedback", "/tf_static"]]
        reverse = {v: k for k, v in TOPICS.items()}
        for conn, t, data in reader.messages(connections=wanted):
            m = reader.deserialize(data, conn.msgtype)
            if conn.topic == LOC:
                q = m.pose.orientation
                rows.append([t, stamp(m), *xyz(m.pose.position), q.x, q.y, q.z, q.w,
                             *xyz(m.velocity.linear), *xyz(m.velocity.angular), m.status,
                             m.latitude, m.longitude, m.altitude])
            elif conn.topic in reverse:
                images[reverse[conn.topic]].append([t, stamp(m)])
            elif conn.topic == "/rock_can/speed_feedback":
                can.append([t, stamp(m), m.speed_kmh, m.speed_cms, m.gear, m.IsValid])
            else:
                for tr in m.transforms:
                    q = tr.transform.rotation
                    tf.append(dict(parent=tr.header.frame_id, child=tr.child_frame_id,
                                   translation=xyz(tr.transform.translation), quaternion_xyzw=[q.x,q.y,q.z,q.w]))
    finally:
        reader.close()
    a = np.asarray(rows)
    # Timestamps are recovered from Python integers, not the float matrix.
    bt = np.array([r[0] for r in rows], dtype=np.int64)
    ht = np.array([r[1] for r in rows], dtype=np.int64)
    if np.any(np.diff(ht) <= 0):
        raise RuntimeError("Localization header timestamps are not strictly increasing")
    quat = a[:, 5:9]
    yaw = np.unwrap(Rotation.from_quat(quat).as_euler("ZYX")[:, 0])
    np.savez_compressed(out / "reference_full.npz", bag_time_ns=bt, time_ns=ht,
                        position_utm_m=a[:,2:5], quaternion_xyzw=quat, yaw_rad=yaw,
                        velocity_linear=a[:,9:12], velocity_angular=a[:,12:15],
                        status=a[:,15].astype(np.int8), latitude_longitude_altitude=a[:,16:19])
    np.savez_compressed(out / "image_index.npz", **{c: np.array(v, dtype=np.int64) for c,v in images.items()})
    np.savez_compressed(out / "can_speed_full.npz",
                        bag_time_ns=np.array([r[0] for r in can],dtype=np.int64),
                        time_ns=np.array([r[1] for r in can],dtype=np.int64),
                        speed_kmh=np.array([r[2] for r in can]), speed_cms=np.array([r[3] for r in can]),
                        gear=np.array([r[4] for r in can]), valid=np.array([r[5] for r in can],dtype=bool))
    metadata.update(reference_source=SOURCE, localization_header_frames="see message_samples.json",
                    tf_static=tf, image_index_columns=["bag_time_ns", "image_header_time_ns"],
                    localization_header_minus_bag_ms=dict(min=float((ht-bt).min()/1e6), max=float((ht-bt).max()/1e6)))
    write_json(out / "inspection.json", metadata)
    logging.info("Inspected %d localization and %s camera messages",len(rows),{c:len(v) for c,v in images.items()})


def candidates(ref, index, start):
    t = (ref["time_ns"] - start)/1e9
    p, yaw = ref["position_utm_m"], ref["yaw_rad"]
    vel = np.gradient(p[:, :2], t, axis=0)
    speed = median_filter(np.linalg.norm(vel, axis=1), size=11)
    moving = speed > 0.18
    # Stop-to-stop blocks; pauses shorter than 3 s remain in the maneuver.
    intervals = []
    edges = np.flatnonzero(np.diff(np.r_[False, moving, False].astype(int)))
    for lo, hi in zip(edges[::2], edges[1::2]):
        if intervals and t[lo] - t[intervals[-1][1]-1] < 3.0:
            intervals[-1][1] = hi
        else:
            intervals.append([int(lo),int(hi)])
    results = []
    for lo, hi in intervals:
        if t[hi-1] - t[lo] < 3:
            continue
        begin, end = t[lo]-3, t[hi-1]+3
        i, j = np.searchsorted(t,[begin,end])
        if j >= len(t):
            continue
        pre, post = speed[i:lo], speed[hi:j+1]
        stationary = begin >= t[0] and len(pre)>0 and len(post)>0 and max(np.quantile(pre,.95), np.quantile(post,.95)) < .18
        signed = (vel[lo:hi]*np.column_stack((np.cos(yaw[lo:hi]),np.sin(yaw[lo:hi])))).sum(axis=1)
        active = speed[lo:hi] > .18
        reverse_fraction = float(np.mean(signed[active] < -.1))
        turn = float(np.rad2deg(yaw[hi-1]-yaw[lo]))
        displacement = float(np.linalg.norm(p[hi-1,:2]-p[lo,:2]))
        coverage = all(index[c][0,1] <= start+begin*1e9 and index[c][-1,1] >= start+end*1e9 for c in CAMS)
        preferred = stationary and coverage and reverse_fraction > .6 and 50 <= abs(turn) <= 135 and 5 <= displacement <= 15
        results.append(dict(id=len(results), start_s=float(begin), end_s=float(end),
                            motion_start_s=float(t[lo]), motion_end_s=float(t[hi-1]),
                            start_time_ns=int(start)+round(begin*1e9), end_time_ns=int(start)+round(end*1e9),
                            duration_s=float(end-begin), yaw_turn_deg=turn, displacement_m=displacement,
                            path_length_m=float(np.linalg.norm(np.diff(p[lo:hi,:2],axis=0),axis=1).sum()),
                            reverse_fraction_from_pose=reverse_fraction, stationary_margins=bool(stationary),
                            cameras_cover_episode=bool(coverage), preferred=bool(preferred),
                            stationary_pre_speed_p95_mps=float(np.quantile(pre,.95)) if len(pre) else None,
                            stationary_post_speed_p95_mps=float(np.quantile(post,.95)) if len(post) else None))
    return results


def ocam_model(path):
    lines = path.read_text().splitlines()
    pol = lines[2].split(); inv = lines[6].split()
    center = list(map(float,lines[10].split())); affine = list(map(float,lines[14].split()))
    height,width = map(int,lines[18].split())
    model = dict(model="Scaramuzza_OCAM_not_Kannala_Brandt",pol=list(map(float,pol[1:])),
                 invpol=list(map(float,inv[1:])), center_row_column=center, affine_c_d_e=affine,
                 height=height,width=width,sf=4.0)
    assert len(model["pol"])==int(pol[0]) and len(model["invpol"])==int(inv[0])
    return model


def ocam_maps(o):
    h,w = o["height"],o["width"]
    row,col = np.indices((h,w), dtype=np.float64)
    row -= h/2; col -= w/2
    norm = np.hypot(row,col)
    theta = np.arctan2(-w/4, norm)
    rho = np.polynomial.polynomial.polyval(theta,o["invpol"])
    scale = np.divide(rho,norm,out=np.zeros_like(norm),where=norm!=0)
    x,y = row*scale,col*scale
    c,d,e = o["affine_c_d_e"]; xc,yc = o["center_row_column"]
    return (x*e+y+yc).astype(np.float32), (x*c+y*d+xc).astype(np.float32)


def extrinsics(spec):
    e = spec["extrinsics"]
    body = np.eye(4)
    body[:3,:3] = Rotation.from_euler("ZYX",[e["yaw"],e["pitch"],e["roll"]],degrees=True).as_matrix()
    body[:3,3] = [e["x"],e["y"],e["z"]]
    optical = np.array([[0,-1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],dtype=float)
    optical_from_vehicle = optical @ np.linalg.inv(body)
    return optical_from_vehicle, np.linalg.inv(optical_from_vehicle)


def interpolate(ref, times, origin, yaw0):
    rt = (ref["time_ns"]-ref["time_ns"][0])/1e9
    qt = (np.asarray(times,dtype=np.int64)-ref["time_ns"][0])/1e9
    if qt.min()<rt[0] or qt.max()>rt[-1]:
        raise RuntimeError("Image time outside localization; no extrapolation allowed")
    pos = np.column_stack([np.interp(qt,rt,ref["position_utm_m"][:,k]) for k in range(3)])
    rot = Slerp(rt,Rotation.from_quat(ref["quaternion_xyzw"]))(qt)
    local_rot = Rotation.from_euler("z",-yaw0)
    lp = local_rot.apply(pos-origin)
    lq = (local_rot*rot).as_quat()
    v = np.column_stack([np.interp(qt,rt,ref["velocity_linear"][:,k]) for k in range(3)])
    nearest = np.searchsorted(ref["time_ns"],times).clip(1,len(rt)-1)
    distance = np.minimum(np.abs(ref["time_ns"][nearest]-times),np.abs(ref["time_ns"][nearest-1]-times))
    gap = ref["time_ns"][nearest]-ref["time_ns"][nearest-1]
    return lp,lq,v,distance,gap


def ffmpeg(path, width, height, fps, log):
    command = ["ffmpeg","-nostdin","-n","-loglevel","warning","-f","rawvideo","-pix_fmt","rgb24",
               "-s",f"{width}x{height}","-r",str(fps),"-i","pipe:0","-an","-c:v","libx264",
               "-preset","fast","-crf","20","-pix_fmt","yuv420p",str(path)]
    return subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)


def preview_candidates(args, out, calibration, choices, index):
    root = out/"candidate_previews"
    root.mkdir(exist_ok=False)
    requests, slots = {}, {}
    maps = {c:ocam_maps(ocam_model(calibration/f"fisheye_undistort/para/{c}/calib_results_{c}.txt")) for c in CAMS}
    for choice in choices:
        if not choice["preferred"]:
            continue
        for n,target in enumerate(np.linspace(choice["start_time_ns"],choice["end_time_ns"],8)):
            slot = (choice["id"],n)
            slots[slot] = {}
            for c in CAMS:
                k = int(np.argmin(np.abs(index[c][:,1]-target)))
                requests.setdefault((c,int(index[c][k,0])),[]).append(slot)
    reader = open_bag(args.bag)
    reverse = {v:k for k,v in TOPICS.items()}
    try:
        for conn,t,data in reader.messages(connections=[c for c in reader.connections if c.topic in reverse]):
            c = reverse[conn.topic]
            if (c,t) not in requests:
                continue
            m = reader.deserialize(data,conn.msgtype)
            image = cv2.imdecode(m.data,cv2.IMREAD_COLOR)
            rect = cv2.resize(cv2.remap(image,*maps[c],interpolation=cv2.INTER_LINEAR),(480,270))
            for slot in requests[(c,t)]:
                slots[slot][c] = rect
    finally:
        reader.close()
    for (candidate,n),images in slots.items():
        if set(images)!=set(CAMS):
            raise RuntimeError("Incomplete candidate preview")
        tile = np.vstack((np.hstack((images["front"],images["back"])),np.hstack((images["left"],images["right"]))))
        cv2.imwrite(str(root/f"candidate_{candidate:02d}_{n:02d}.jpg"),tile)
    write_json(root/"origin.json",dict(source=SOURCE,candidates=[c for c in choices if c["preferred"]],
                                    purpose="Visual static-scene screening; approximate evenly spaced timestamps, not synchronized DGGT windows"))


def prepare(args, out, calibration):
    inspection = json.loads((out/"inspection.json").read_text())
    ref = np.load(out/"reference_full.npz")
    index = np.load(out/"image_index.npz")
    choices = candidates(ref,index,inspection["start_time_ns"])
    if not (out/"candidates.json").exists():
        write_json(out/"candidates.json",dict(criteria="pose-derived speed >0.18 m/s; merge pauses <3s; require 3s stationary margins; prefer reverse fraction >0.6, yaw 50-135 deg, displacement 5-15m", candidates=choices))
    logging.info("Candidate maneuvers: %s", json.dumps(choices))
    if args.preview_candidates:
        preview_candidates(args,out,calibration,choices,index)
        return
    if args.inspect_only:
        return
    if args.candidate is None:
        eligible = [c for c in choices if c["preferred"]]
        if not eligible:
            raise RuntimeError("No complete preferred maneuver; inspect candidates.json, do not fabricate")
        selected = min(eligible,key=lambda c:abs(c["duration_s"]-30))
    else:
        selected = choices[args.candidate]
    if not selected["stationary_margins"] or not selected["cameras_cover_episode"]:
        raise RuntimeError("Selected maneuver lacks stationary margins or four-camera coverage")
    episode = out/"episode"
    episode.mkdir(exist_ok=False)
    start,end = selected["start_time_ns"],selected["end_time_ns"]
    fidx = np.flatnonzero((index["front"][:,1]>=start)&(index["front"][:,1]<=end))
    ft = index["front"][fidx,1]
    matched, residual = {}, {}
    for c in CAMS:
        ct = index[c][:,1]
        if np.any(np.diff(ct)<=0):
            raise RuntimeError(f"Nonmonotonic image times: {c}")
        k = np.searchsorted(ct,ft).clip(1,len(ct)-1)
        k -= (np.abs(ct[k-1]-ft)<=np.abs(ct[k]-ft)).astype(int)
        matched[c] = k; residual[c] = ct[k]-ft
    valid = np.logical_and.reduce([np.abs(residual[c])<=60_000_000 for c in CAMS])
    if not np.all(valid):
        raise RuntimeError(f"Incomplete synchronization: {int((~valid).sum())}/{len(ft)} front frames exceed 60ms")
    if len(ft)<8 or np.max(np.diff(ft))>500_000_000:
        raise RuntimeError("Incomplete image episode (fewer than 8 frames or >500ms gap)")
    origin = np.array([np.interp((start-ref["time_ns"][0])/1e9,(ref["time_ns"]-ref["time_ns"][0])/1e9,ref["position_utm_m"][:,k]) for k in range(3)])
    yaw0 = float(np.interp((start-ref["time_ns"][0])/1e9,(ref["time_ns"]-ref["time_ns"][0])/1e9,ref["yaw_rad"]))
    mask = (ref["time_ns"]>=start)&(ref["time_ns"]<=end)
    lp,lq,v,dist,gap = interpolate(ref,ref["time_ns"][mask],origin,yaw0)
    np.savez_compressed(episode/"reference_selected.npz",time_ns=ref["time_ns"][mask],
                        position_local_m=lp, quaternion_local_xyzw=lq,
                        yaw_local_rad=ref["yaw_rad"][mask]-yaw0,
                        position_utm_m=ref["position_utm_m"][mask], quaternion_xyzw=ref["quaternion_xyzw"][mask],
                        velocity_linear=v)
    width,height = args.width,args.height
    config = json.loads((calibration/"utils/camera_config_right_hand.json").read_text())
    maps, camera_info, required, sequences = {}, {}, {}, {}
    for c in CAMS:
        for d in (episode/"raw"/c,episode/"rectified"/c):
            d.mkdir(parents=True)
        model = ocam_model(calibration/f"fisheye_undistort/para/{c}/calib_results_{c}.txt")
        maps[c] = ocam_maps(model)
        native_k = np.array([[model["width"]/4,0,model["width"]/2],[0,model["width"]/4,model["height"]/2],[0,0,1]])
        scaled_k = np.diag([width/model["width"],height/model["height"],1])@native_k
        spec = config["CAM_REAR" if c=="back" else "CAM_"+c.upper()]
        ov,vo = extrinsics(spec)
        camera_info[c] = dict(topic=TOPICS[c],ocam=model,K_native=native_k.tolist(),K_scaled=scaled_k.tolist(),
                              output_size=[width,height], official_config=spec,
                              T_optical_from_vehicle=ov.tolist(),T_vehicle_from_optical=vo.tolist(),
                              raw_dir=f"episode/raw/{c}",rectified_dir=f"episode/rectified/{c}",
                              official_fov127_K_note="Official rounded 127 degree FOV differs slightly from exact sf=4; K_scaled here uses exact OCAM LUT focal width/4.")
        all_k = np.union1d(matched[c],np.flatnonzero((index[c][:,1]>=start)&(index[c][:,1]<=end)))
        required[c] = {int(index[c][k,0]): int(index[c][k,1]) for k in all_k}
        sequences[c] = [dict(bag_time_ns=int(index[c][k,0]),image_time_ns=int(index[c][k,1]),
                             raw=f"episode/raw/{c}/{int(index[c][k,1])}.jpg",
                             rectified=f"episode/rectified/{c}/{int(index[c][k,1])}.jpg") for k in all_k]
    frames = []
    arrays = {}
    for c in CAMS:
        times = index[c][matched[c],1]
        lp,lq,v,dist,gap = interpolate(ref,times,origin,yaw0)
        arrays.update({f"{c}_image_time_ns":times, f"{c}_position_local_m":lp,
                       f"{c}_quaternion_local_xyzw":lq,f"{c}_velocity_linear":v,
                       f"{c}_sync_residual_ns":residual[c],f"{c}_reference_nearest_distance_ns":dist,
                       f"{c}_reference_bracket_gap_ns":gap})
    np.savez_compressed(episode/"frame_poses.npz",front_time_ns=ft,**arrays)
    for n,t in enumerate(ft):
        observations = {}
        for c in CAMS:
            ct = int(arrays[f"{c}_image_time_ns"][n])
            observations[c] = dict(image_time_ns=ct,bag_time_ns=int(index[c][matched[c][n],0]),
                                   sync_residual_ns=int(residual[c][n]),
                                   raw=f"episode/raw/{c}/{ct}.jpg",rectified=f"episode/rectified/{c}/{ct}.jpg",
                                   position_local_m=arrays[f"{c}_position_local_m"][n].tolist(),
                                   quaternion_local_xyzw=arrays[f"{c}_quaternion_local_xyzw"][n].tolist())
        frames.append(dict(index=n,front_time_ns=int(t),elapsed_s=float((t-start)/1e9),cameras=observations))
    logging.info("Extracting candidate %d: %.3f..%.3fs, %d synchronized frames",selected["id"],selected["start_s"],selected["end_s"],len(ft))
    reader = open_bag(args.bag)
    seen = {c:set() for c in CAMS}
    reverse = {v:k for k,v in TOPICS.items()}
    try:
        all_bt = [t for c in CAMS for t in required[c]]
        conns = [c for c in reader.connections if c.topic in reverse]
        for conn,t,data in reader.messages(connections=conns,start=min(all_bt),stop=max(all_bt)+1):
            c = reverse[conn.topic]
            if t not in required[c]:
                continue
            m = reader.deserialize(data,conn.msgtype); it = required[c][t]
            image = cv2.imdecode(m.data,cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (camera_info[c]["ocam"]["height"],camera_info[c]["ocam"]["width"]):
                raise RuntimeError(f"Bad native image dimensions: {c} {it}")
            with (episode/"raw"/c/f"{it}.jpg").open("xb") as f:
                f.write(m.data.tobytes())
            rect = cv2.remap(image,*maps[c],interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            rect = cv2.resize(rect,(width,height),interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(episode/"rectified"/c/f"{it}.jpg"),rect,[cv2.IMWRITE_JPEG_QUALITY,92]):
                raise RuntimeError("JPEG write failed")
            seen[c].add(t)
    finally:
        reader.close()
    if any(seen[c]!=set(required[c]) for c in CAMS):
        raise RuntimeError("Missing compressed images during extraction")
    fps = float(1e9/np.median(np.diff(ft)))
    logpath = episode/"ffmpeg.log"
    with logpath.open("xb") as log:
        for c in CAMS:
            cam_times = np.array([s["image_time_ns"] for s in sequences[c]],dtype=np.int64)
            cam_fps = float(1e9/np.median(np.diff(cam_times)))
            encoder = ffmpeg(episode/f"{c}.mp4",width,height,cam_fps,log)
            try:
                for s in sequences[c]:
                    rgb = cv2.cvtColor(cv2.imread(str(out/s["rectified"])),cv2.COLOR_BGR2RGB)
                    encoder.stdin.write(rgb.tobytes())
            finally:
                encoder.stdin.close()
                code = encoder.wait()
            if code:
                raise RuntimeError(f"ffmpeg {c} failed: {code}")
        quad = ffmpeg(episode/"quad.mp4",width*2,height*2,fps,log)
        try:
            for frame in frames:
                rgb = {}
                for c in CAMS:
                    rgb[c] = cv2.cvtColor(cv2.imread(str(out/frame["cameras"][c]["rectified"])),cv2.COLOR_BGR2RGB)
                tile = np.vstack((np.hstack((rgb["front"],rgb["back"])),np.hstack((rgb["left"],rgb["right"]))))
                quad.stdin.write(tile.tobytes())
        finally:
            quad.stdin.close()
            code = quad.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed {code}; see {logpath}")
    windows = [frames[int(i)] for i in np.linspace(0,len(frames)-1,8).round().astype(int)]
    episode.joinpath("sampled_quads").mkdir()
    for n,frame in enumerate(windows):
        views = [cv2.imread(str(out/frame["cameras"][c]["rectified"])) for c in CAMS]
        tile = np.vstack((np.hstack(views[:2]),np.hstack(views[2:])))
        cv2.imwrite(str(episode/"sampled_quads"/f"{n:02d}.jpg"),tile)
    can = np.load(out/"can_speed_full.npz")
    cm = (can["time_ns"]>=start)&(can["time_ns"]<=end)
    gears,counts = np.unique(can["gear"][cm],return_counts=True)
    manifest = dict(schema_version=1,source=SOURCE,bridge="otheragentfeedforward",bag=inspection,
                    upstream_commit=COMMIT,selected_episode=selected,
                    message_samples="message_samples.json",
                    selection_basis="Observed localization pose-derived speed/direction/yaw and stationary margins; no planner or model fitting. Static scene requires visual review.",
                    local_frame=dict(origin_utm_m=origin.tolist(),anchor_time_ns=start,anchor_yaw_rad=yaw0,
                                     convention="meters; rotate UTM displacement by Rz(-anchor_yaw); start ego yaw zero; preserve roll/pitch",
                                     raw_origin_assumptions="Message definition says UTM, but frame_id is empty and recorded coordinates are small/local-looking, not verified absolute UTM. The position_utm_m key preserves schema terminology, not a confirmed geodetic frame. Zone/datum, upstream rebase and physical localization origin are unspecified. No verified rear-axle reference or localization-to-official-vehicle-origin transform. Do not treat camera/pose origin coincidence as calibrated."),
                    extrinsic_convention="T_optical_from_vehicle maps vehicle points to optical x-right,y-down,z-forward. Inverse from official camera_utils + pose_utils: T_vehicle_from_optical = [Rz(yaw)Ry(pitch)Rx(roll),t] @ inverse(cam2pixel). Vehicle x-forward,y-left,z-up. Localization origin relationship unverified.",
                    cameras=camera_info,frames=frames,camera_sequences=sequences,sampled_windows=windows,
                    synchronization=dict(max_allowed_ns=60_000_000,max_abs_camera_residual_ns=max(int(np.abs(residual[c]).max()) for c in CAMS),
                                         max_reference_nearest_distance_ns=max(int(arrays[f"{c}_reference_nearest_distance_ns"].max()) for c in CAMS),
                                         max_reference_bracket_gap_ns=max(int(arrays[f"{c}_reference_bracket_gap_ns"].max()) for c in CAMS),
                                         max_front_frame_gap_ns=int(np.diff(ft).max()),interpolation="linear position/velocity and quaternion SLERP at each camera header time; no extrapolation"),
                    can_gear_counts={str(g):int(n) for g,n in zip(gears,counts)},
                    can_speed_valid_count=int(np.count_nonzero(can["valid"][cm])),
                    can_gear_semantics="Raw enum, not inferred without documentation; reversing identified by pose displacement projected on yaw. CAN validity flag must be respected; speed is not independent verification.",
                    videos=dict(fps=fps,encoding="ffmpeg rgb24 input -> H264 yuv420p",timing="constant median cadence preview; manifest nanoseconds are authoritative",quad_order=[["front","back"],["left","right"]]),
                    array_schema=dict(reference_full="time_ns int64[N], bag_time_ns int64[N], position_utm_m float64[N,3], quaternion_xyzw float64[N,4], yaw_rad unwrapped[N], velocity_linear/velocity_angular[N,3] raw message components (frame unspecified), status int8[N], latitude_longitude_altitude[N,3]",
                                      reference_selected="time_ns[M], position_local_m[M,3], quaternion_local_xyzw[M,4], yaw_local_rad[M], position_utm_m[M,3], quaternion_xyzw[M,4], velocity_linear[M,3] (raw frame)",
                                      frame_poses="front_time_ns[F]; per-camera image_time_ns[F], position_local_m[F,3], quaternion_local_xyzw[F,4], velocity_linear[F,3], sync_residual_ns[F] (image-front), reference_nearest_distance_ns[F], reference_bracket_gap_ns[F]"))
    if args.selection_review:
        review = json.loads(args.selection_review.read_text())
        if review["selected_candidate_id"] != selected["id"]:
            raise RuntimeError("Visual review does not match selected candidate")
        manifest["visual_selection_review"] = review
    write_json(out/"manifest.json",manifest)
    write_json(out/"sampled_windows.json",dict(source=SOURCE,windows=windows,cameras=camera_info))
    write_json(out/"preparation_report.json",dict(status="complete",source=SOURCE,selected_episode=selected,
               candidates=choices,synchronization=manifest["synchronization"],frame_count=len(frames),
               unique_camera_frames={c:len(seen[c]) for c in CAMS},can_gear_counts=manifest["can_gear_counts"],
               limitations=["Recorded localization is an estimate, not independent GT.","Static scene has not been algorithmically certified.","Localization/vehicle physical reference origins not documented; no rear-axle claim."]))
    logging.info("Complete: %s",out/"manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag",type=Path,default=Path("data/raw/parkinge2e/demo_bag.bag"))
    parser.add_argument("--output",type=Path,default=Path("outputs/parkinge2e_20261010_01/preparation"))
    parser.add_argument("--calibration",type=Path,default=Path("data/raw/parkinge2e/calibration"))
    parser.add_argument("--inspect-only",action="store_true")
    parser.add_argument("--preview-candidates",action="store_true")
    parser.add_argument("--inspection-dir",type=Path,help="Reuse a prior inspection's immutable reference/index artifacts")
    parser.add_argument("--selection-review",type=Path,help="Include explicit human/agent visual screening provenance")
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--candidate",type=int)
    parser.add_argument("--width",type=int,default=640)
    parser.add_argument("--height",type=int,default=360)
    args = parser.parse_args()
    if args.verify_only:
        verify(args.output)
        return
    if (args.width,args.height) not in [(640,360),(480,270)]:
        parser.error("Use calibrated aspect ratio: 640x360 or 480x270")
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.preview_candidates and ((args.output/"manifest.json").exists() or (args.output/"episode").exists()):
        parser.error("Output already contains an episode; refusing to overwrite")
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),logging.FileHandler(args.output/"preparation.log",mode="a")])
    try:
        download_calibration(args.calibration)
        if args.inspection_dir:
            import shutil
            for name in ["inspection.json","message_samples.json","reference_full.npz","image_index.npz","can_speed_full.npz"]:
                target = args.output/name
                if target.exists():
                    raise RuntimeError(f"Refusing to overwrite {target}")
                shutil.copyfile(args.inspection_dir/name,target)
        else:
            inspect(args.bag,args.output)
        prepare(args,args.output,args.calibration)
    except Exception:
        logging.exception("Preparation incomplete; artifacts preserved, no synthetic data substituted")
        raise


def verify(out):
    manifest = json.loads((out/"manifest.json").read_text())
    full = np.load(out/"reference_full.npz")
    poses = np.load(out/"episode/frame_poses.npz")
    assert len(full["time_ns"]) == 21256
    assert full["time_ns"].dtype == np.int64
    assert len(manifest["sampled_windows"]) == 8
    assert np.all(np.diff(full["time_ns"])>0)
    origin = np.array(manifest["local_frame"]["origin_utm_m"])
    yaw0 = manifest["local_frame"]["anchor_yaw_rad"]
    videos = {}
    for c in CAMS:
        info = manifest["cameras"][c]
        np.testing.assert_allclose(np.array(info["T_optical_from_vehicle"])@np.array(info["T_vehicle_from_optical"]),np.eye(4),atol=1e-12)
        assert np.max(np.abs(poses[f"{c}_sync_residual_ns"])) <= 60_000_000
        p,q,_,_,_ = interpolate(full,poses[f"{c}_image_time_ns"],origin,yaw0)
        np.testing.assert_allclose(p,poses[f"{c}_position_local_m"],atol=1e-12)
        np.testing.assert_allclose(q,poses[f"{c}_quaternion_local_xyzw"],atol=1e-12)
        for s in manifest["camera_sequences"][c]:
            raw = cv2.imread(str(out/s["raw"])); rect = cv2.imread(str(out/s["rectified"]))
            assert raw.shape[:2] == (1080,1920)
            assert rect.shape[:2] == tuple(info["output_size"][::-1])
    for c in [*CAMS,"quad"]:
        result = subprocess.run(["ffprobe","-v","error","-count_frames","-select_streams","v:0",
                                 "-show_entries","stream=width,height,pix_fmt,nb_read_frames,codec_name,duration",
                                 "-of","json",str(out/"episode"/f"{c}.mp4")],check=True,capture_output=True,text=True)
        stream = json.loads(result.stdout)["streams"][0]
        expected = len(manifest["frames"]) if c=="quad" else len(manifest["camera_sequences"][c])
        assert int(stream["nb_read_frames"]) == expected
        assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
        videos[c] = stream
    tests = subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-p","test_parkinge2e.py","-v"],capture_output=True,text=True)
    result = dict(status="passed" if tests.returncode==0 else "failed",source=SOURCE,
                  full_reference_count=len(full["time_ns"]),frame_count=len(manifest["frames"]),
                  synchronization=manifest["synchronization"],videos=videos,unit_test_stdout=tests.stdout,
                  unit_test_stderr=tests.stderr,unit_test_returncode=tests.returncode,
                  verified="All saved JPEG dimensions, all video decoded frame counts, transform inverses, exact int64 timestamps and recomputed interpolated camera poses")
    write_json(out/"verification.json",result)
    print(json.dumps(result,indent=2))
    if tests.returncode:
        raise RuntimeError("Preparation unit tests failed")


if __name__ == "__main__":
    main()
