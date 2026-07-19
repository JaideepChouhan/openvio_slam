"""
persistence/map_serializer.py
==============================
Save and load the complete map to/from a NumPy .npz archive.

Typical file size: ~30 MB for a 100-room map (500 KFs, 5000 MPs).
Format version: openvio_slam_v1
"""

import numpy as np
import json
import os
import time
from typing import Tuple


MAP_VERSION = "openvio_slam_v1"


def save_map(save_path: str, keyframes: dict, map_points: dict, metadata: dict) -> str:
    save_path = os.path.expanduser(save_path)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    kf_list = list(keyframes.values())
    kf_ids    = np.array([kf.id for kf in kf_list], dtype=np.int64)
    kf_poses  = np.array([np.concatenate([kf.q_GC, kf.p_GC]) for kf in kf_list], dtype=np.float64)
    kf_feat_counts = np.array([len(kf.descriptors) for kf in kf_list], dtype=np.int32)

    valid_kfs = [kf for kf in kf_list if len(kf.descriptors) > 0]
    if valid_kfs:
        all_descs = np.vstack([kf.descriptors for kf in valid_kfs])
        all_npts  = np.vstack([kf.norm_pts    for kf in valid_kfs])
    else:
        all_descs = np.zeros((0, 32), dtype=np.uint8)
        all_npts  = np.zeros((0, 2),  dtype=np.float32)

    mp_list   = list(map_points.values())
    mp_ids    = np.array([mp.id for mp in mp_list], dtype=np.int64)
    mp_xyz    = np.array([mp.position   for mp in mp_list], dtype=np.float64) if mp_list else np.zeros((0,3))
    mp_descs  = np.array([mp.descriptor for mp in mp_list], dtype=np.uint8)  if mp_list else np.zeros((0,32), dtype=np.uint8)
    mp_stats  = np.array([[mp.n_observed, mp.n_visible] for mp in mp_list], dtype=np.int32) if mp_list else np.zeros((0,2), dtype=np.int32)

    meta = {'version': MAP_VERSION,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'n_keyframes': len(kf_list),
            'n_map_points': len(mp_list), **metadata}

    np.savez_compressed(
        save_path,
        kf_ids=kf_ids, kf_poses=kf_poses,
        kf_descriptors=all_descs, kf_norm_pts=all_npts,
        kf_feat_counts=kf_feat_counts,
        mp_ids=mp_ids, mp_xyz=mp_xyz,
        mp_descriptors=mp_descs, mp_stats=mp_stats,
        metadata_json=np.array([json.dumps(meta)]),
    )

    actual_path = save_path if save_path.endswith('.npz') else save_path + '.npz'
    size_mb = os.path.getsize(actual_path) / 1e6
    print(f"Map saved: {actual_path} ({len(kf_list)} KFs, {len(mp_list)} MPs, {size_mb:.1f} MB)")
    return actual_path


def load_map(load_path: str) -> Tuple[list, list, dict]:
    load_path = os.path.expanduser(load_path)
    if not load_path.endswith('.npz'):
        load_path += '.npz'

    data = np.load(load_path, allow_pickle=False)
    meta = json.loads(str(data['metadata_json'][0]))
    print(f"Loading map: {meta.get('timestamp','?')} "
          f"({meta.get('n_keyframes','?')} KFs, {meta.get('n_map_points','?')} MPs)")

    kf_ids, kf_poses = data['kf_ids'], data['kf_poses']
    all_descs, all_npts = data['kf_descriptors'], data['kf_norm_pts']
    kf_counts = data['kf_feat_counts']

    keyframes_data, offset = [], 0
    for i, kf_id in enumerate(kf_ids):
        n = int(kf_counts[i])
        keyframes_data.append({
            'id': int(kf_id), 'q_GC': kf_poses[i,:4], 'p_GC': kf_poses[i,4:],
            'descriptors': all_descs[offset:offset+n],
            'norm_pts':    all_npts[offset:offset+n],
        })
        offset += n

    map_points_data = [
        {'id': int(mp_ids[i]), 'position': data['mp_xyz'][i],
         'descriptor': data['mp_descriptors'][i],
         'n_observed': int(data['mp_stats'][i,0]),
         'n_visible':  int(data['mp_stats'][i,1])}
        for i in range(len(data['mp_ids']))
    ]

    return keyframes_data, map_points_data, meta
