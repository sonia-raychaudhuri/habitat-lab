#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import sys
sys.path.insert(0, "")    
import pickle
from PIL import Image 
import imageio

import numpy as np
import torch

from habitat_baselines.config.default import get_config

import habitat
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
from habitat.utils.visualizations import maps

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

COORDINATE_EPSILON = 1e-6
COORDINATE_MIN = -120.3241 - COORDINATE_EPSILON
# COORDINATE_MAX = 90.0399 + COORDINATE_EPSILON
COORDINATE_MAX = 120.0399 + COORDINATE_EPSILON


def to_grid(
    realworld_x,
    realworld_y,
    coordinate_min,
    coordinate_max,
    grid_resolution
):
    r"""Return gridworld index of realworld coordinates assuming top-left corner
    is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max)
    """
    grid_size = (
        (coordinate_max - coordinate_min) / grid_resolution[0],
        (coordinate_max - coordinate_min) / grid_resolution[1],
    )
    grid_x = int((coordinate_max - realworld_x) / grid_size[0])
    grid_y = int((realworld_y - coordinate_min) / grid_size[1])
    return grid_x, grid_y



mapDict={}

def example_get_topdown_map():
    config = habitat.get_config("configs/tasks/vln_rxr.yaml")
    dataset = habitat.make_dataset(
        id_dataset=config.DATASET.TYPE, config=config.DATASET
    )
    env = habitat.Env(config=config, dataset=dataset)
    
    count = 0
    while(len(mapDict)<95):
        env.reset()
        if env.current_episode.scene_id in mapDict: 
            continue
        top_down_map = maps.get_topdown_map(env.sim, map_resolution=(300, 300), num_samples=20000, draw_border=False)
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0,0,0]], dtype=np.uint8
                )
        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]
        # padding = int(np.ceil(top_down_map.shape[0] / 125))
        padding = int(np.ceil(top_down_map.shape[0] / 400))

        range_x = (
            max(range_x[0] - padding, 0),
            min(range_x[-1] + padding + 1, top_down_map.shape[0]),
        )
        range_y = (
            max(range_y[0] - padding, 0),
            min(range_y[-1] + padding + 1, top_down_map.shape[1]),
        )

        # top_down_map = top_down_map[
        #     range_x[0] : range_x[1], range_y[0] : range_y[1]
        # ]
        # top_down_map = recolor_map[top_down_map][:,:,0]

        top_down_map[range_x[0] : range_x[1], range_y[0] : range_y[1]] += 1

        semMap = np.zeros((300,300,3))
        semMap[:,:,0] = top_down_map
        # ax, ay = to_grid(env.sim.get_agent_state().position[0], env.sim.get_agent_state().position[2], COORDINATE_MIN, COORDINATE_MAX, (5000,5000))
        # patch = top_down_map[ax-100:ax+100, ay-100:ay+100]
        count += 1
        print ("Serial: ", count)

        mapDict[env.current_episode.scene_id]=semMap


        # np.save('3.npy',  mapDict)   
        # import json
        # json.dump(mapDict, file('3.txt', 'w'))


        with open('map300.pickle', 'wb') as handle:
            pickle.dump(mapDict, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        # with open('mapDict.pickle', 'rb') as handle:
        #     b = pickle.load(handle)


def main():
    example_get_topdown_map()
    print("fin")

if __name__ == "__main__":
    main()
