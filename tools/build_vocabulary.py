#!/usr/bin/env python3
"""
tools/build_vocabulary.py
=========================
Offline tool: build and save an ORB Bag-of-Words vocabulary.

Usage:
    python3 tools/build_vocabulary.py --images /path/to/images/ \
                                      --output config/orb_vocabulary.pkl \
                                      [--k1 10] [--k2 10]

Training images should be diverse indoor images. Even 100 images gives
a usable vocabulary for single-room mapping. Expected runtime: ~2 min
for 500 images.
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from openvio_slam.loop_closure.vocabulary import ORBVocabulary


def extract_orb_descriptors(image_paths: list, max_per_image: int = 500) -> np.ndarray:
    orb = cv2.ORB_create(nfeatures=max_per_image)
    all_descs = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping: {img_path}"); continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, descs = orb.detectAndCompute(gray, None)
        if descs is not None and len(descs) > 0:
            all_descs.append(descs)
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(image_paths)} images, "
                  f"{sum(len(d) for d in all_descs)} descriptors")
    if not all_descs:
        raise RuntimeError("No descriptors extracted — check image paths")
    return np.vstack(all_descs)


def main():
    parser = argparse.ArgumentParser(description="Build ORB BoW vocabulary")
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', default='config/orb_vocabulary.pkl')
    parser.add_argument('--k1', type=int, default=10)
    parser.add_argument('--k2', type=int, default=10)
    parser.add_argument('--max-images', type=int, default=None)
    args = parser.parse_args()

    img_dir = Path(args.images)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = sorted([p for p in img_dir.rglob('*') if p.suffix.lower() in exts])
    if not image_paths:
        print(f"ERROR: No images found in {img_dir}"); sys.exit(1)

    if args.max_images:
        step = max(1, len(image_paths) // args.max_images)
        image_paths = image_paths[::step]

    print(f"Training vocabulary: {len(image_paths)} images, "
          f"k1={args.k1} k2={args.k2} → {args.k1*args.k2} words")

    all_descs = extract_orb_descriptors(image_paths)
    print(f"Total descriptors: {len(all_descs)}")

    vocab = ORBVocabulary(k1=args.k1, k2=args.k2)
    vocab.train(all_descs, n_training_images=len(image_paths))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    vocab.save(args.output)
    print(f"\nDone! Vocabulary saved to: {args.output}")


if __name__ == '__main__':
    main()
