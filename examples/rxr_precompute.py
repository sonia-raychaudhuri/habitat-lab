from typing import Any, Union, List

import argparse
import numpy as np
from gym import spaces
import gzip
import json
import math
import h5py
import tqdm

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps

GT_PATH = "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"
TOPDOWN_MAP_PATH = "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_occupancy.h5"
SUCCESS_DISTANCE = 3.0
split = "val_seen"
role = "guide"
map_resolution = 1024
TOPDOWN_MAPS = {}
NUMBER_OF_EPISODES = 10

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def load_gt_actions(episode):
    gt_json = {}
    with gzip.open(
        GT_PATH.format(split=split, role=role), "rt"
    ) as f:
        gt_json.update(json.load(f))

    actions = gt_json[str(episode.episode_id)]["actions"]
    str(actions)
    return actions

def run_interactive_gt(episode_id):
    config = habitat.get_config("configs/tasks/vln_rxr.yaml")

    
    config.defrost()

    if episode_id != "all":
        config.DATASET.EPISODES_ALLOWED = [episode_id]
    else:
        config.DATASET.EPISODES_ALLOWED = None

    config.freeze()

    env = habitat.Env(
        config=config
    )

    print("Environment creation successful")
    
    hout = h5py.File(TOPDOWN_MAP_PATH.format(split=split,role=role), 'w')

    count_episode = 0
    total_episodes = min(env.number_of_episodes, NUMBER_OF_EPISODES)
    pbar = tqdm.tqdm(total=total_episodes)

    while count_episode < total_episodes:
        observations = env.reset()
        episode = env.current_episode
        
        print(f"Episode={episode.episode_id}.\nInstruction text: [{observations['instruction']['text']}]")

        actions = load_gt_actions(episode)

        lower_bound, upper_bound = env.sim.pathfinder.get_bounds()
        meters_per_pixel =  min(
            abs(upper_bound[coord] - lower_bound[coord]) / map_resolution
            for coord in [0, 2]
        )

        count_steps = 0
        for action in actions:
            agent_position = env.sim.get_agent(0).state.position
            height = agent_position[1]
            
            if episode.scene_id not in hout or str(height) not in hout[episode.scene_id]:
                top_down_map = env.sim.pathfinder.get_topdown_view(meters_per_pixel, height).astype(np.uint8)
                
                if episode.scene_id not in hout:
                    hout.create_group(episode.scene_id)
                
                hout[episode.scene_id][str(height)] = top_down_map
            
            observations = env.step(action)
            count_steps += 1

        print(f"Episode finished after {count_steps} steps.")
        count_episode += 1
        pbar.update()
        

def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Habitat API Demo")

    parser.add_argument(
        "--episode_id",
        default="all",
        help="Enter Episode Id",
    )

    args = parser.parse_args()

    run_interactive_gt(args.episode_id)
    