#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser

cur_dir = os.path.dirname(__file__)

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

mipnerf360_outdoor_cap = [6131954, 3636448, 5834784, 4961797, 3783761]
mipnerf360_indoor_cap = [1593376, 1222956, 1852335, 1244819]
tanks_and_temples_cap = [2541226, 1026508]
deep_blending_cap = [3405153, 2546116]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
    parser.add_argument("--tanksandtemples", "-tat", required=False, type=str)
    parser.add_argument("--deepblending", "-db", required=False, type=str)
    parser.add_argument('--config', required=True, type=str)
    
    args = parser.parse_args()

if not args.skip_training:
    common_args = f"--splatting_config=\"{args.config}\" --eval --test_iterations -1 --save_iterations -1"
    for scene, cap in zip(mipnerf360_outdoor_scenes, mipnerf360_outdoor_cap):
        source = args.mipnerf360 + "/" + scene
        command = f"python train.py -s {source} -i images_4 -m {args.output_path}/{scene} {common_args} --cap_max {cap}"
        os.system(command)
    for scene, cap in zip(mipnerf360_indoor_scenes, mipnerf360_indoor_cap):
        source = args.mipnerf360 + "/" + scene
        command = f"python train.py -s {source} -i images_2 -m {args.output_path}/{scene} {common_args} --cap_max {cap}"
        os.system(command)
    for scene, cap in zip(tanks_and_temples_scenes, tanks_and_temples_cap):
        source = args.tanksandtemples + "/" + scene
        command = f"python train.py -s {source} -m {args.output_path}/{scene} {common_args} --cap_max {cap}"
        os.system(command)
    for scene, cap in zip(deep_blending_scenes, deep_blending_cap):
        source = args.deepblending + "/" + scene
        command = f"python train.py -s {source} -m {args.output_path}/{scene} {common_args} --cap_max {cap}"
        os.system(command)

if not args.skip_rendering:
    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(args.deepblending + "/" + scene)

    common_args = " --eval --skip_train --data_device cuda"
    for scene, source in zip(all_scenes, all_sources):
        for it in [30000]:
            os.system(f"python render.py --iteration {it} -s {source} -m {args.output_path}/{scene} {common_args}")

if not args.skip_metrics:
    for scene in all_scenes:
        scenes_string = "\"" + args.output_path + "/" + scene + "\""

        os.system("python metrics.py -m " + scenes_string)
