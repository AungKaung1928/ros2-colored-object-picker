#!/usr/bin/env python3
"""Run ColorDetector on a synthetic 640x480 frame (grey desk, 3 coloured squares) and save the
annotated output to docs/media/demo.png. No ROS, no camera. Run from the repo root: PYTHONPATH=. python3 scripts/make_demo.py"""
import cv2
import numpy as np
import yaml

from colored_object_picker.color_detector import ColorDetector, load_color_configs_from_dict

with open('config/colors.yaml') as f:
    detector = ColorDetector(load_color_configs_from_dict(yaml.safe_load(f)))

frame = np.full((480, 640, 3), 90, np.uint8)
blocks = {(0, 220, 255): (80, 80), (230, 80, 20): (440, 300), (40, 200, 40): (120, 300)}
for bgr, (x, y) in blocks.items():  # yellow, blue, green (BGR)
    cv2.rectangle(frame, (x, y), (x + 90, y + 90), bgr, -1)

objects, _ = detector.detect_objects(frame)
for o in objects:
    print(o.color_name, o.centroid, o.area)
cv2.imwrite('docs/media/demo.png', detector.draw_detections(frame.copy(), objects))
