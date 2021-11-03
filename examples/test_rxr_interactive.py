from typing import Any, Union, List

import argparse
import numpy as np
from gym import spaces
import gzip
import json
import math
from dtw import dtw
from fastdtw import fastdtw

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
from habitat.tasks.utils import (
    cartesian_to_polar,
)
from habitat.utils.geometry_utils import quaternion_rotate_vector
from scipy import ndimage
import cv2


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
NEXT_EPISODE="n"

PREDICTION_PATH = "data/datasets/RxR_VLNCE_v0/leaderboard/{split}/predicted_actions_0_{split}.json"
GT_PATH = "data/datasets/RxR_VLNCE_v0/{split}/{split}_{role}_gt.json.gz"
SUCCESS_DISTANCE = 3.0
split = "val_seen"
role = "guide"
egocentric_map_size = 200
map_size_floor = math.floor(egocentric_map_size/2)
map_size_ceil = math.ceil(egocentric_map_size/2)
map_resolution = 500

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

def load_predicted_actions(episode):
    pred_json = {}
    with open(PREDICTION_PATH.format(split=split, role=role), "r") as f:
        pred_json.update(json.load(f))

    actions = pred_json[str(episode.episode_id)]["actions"]
    str(actions)
    return actions

def _quat_to_xy_heading(ref_rotation):
    heading_vector = quaternion_rotate_vector(
        ref_rotation.inverse(), np.array([0, 0, -1])
    )

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    z_neg_z_flip = np.pi
    return np.array(phi) + z_neg_z_flip
    
class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

@habitat.registry.register_sensor(name="agent_position")
class AgentPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "agent_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return self._sim.get_agent_state().position

@habitat.registry.register_sensor(name="agent_start_position")
class AgentStartPositionSensor(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "agent_start_position"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return episode.start_position

@habitat.registry.register_sensor(name="episode")
class NavEpisode(habitat.Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)

        self._sim = sim

    # Defines the name of the sensor in the sensor suite dictionary
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "episode"

    # Defines the type of the sensor
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return habitat.SensorTypes.MEASUREMENT

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    # This is called whenver reset is called or an action is taken
    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return episode

@habitat.registry.register_measure
class NDTW(habitat.Measure):
    r"""NDTW (Normalized Dynamic Time Warping)

    ref: Effective and General Evaluation for Instruction
        Conditioned Navigation using Dynamic Time
        Warping - Magalhaes et. al
    https://arxiv.org/pdf/1907.05446.pdf
    """

    cls_uuid: str = "ndtw"

    @staticmethod
    def euclidean_distance(
        position_a: Union[List[float], np.ndarray],
        position_b: Union[List[float], np.ndarray],
    ) -> float:
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def __init__(
        self, sim, config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        self.gt_json = {}
        with gzip.open(
            GT_PATH.format(split=split, role=role), "rt"
        ) as f:
            self.gt_json.update(json.load(f))

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.locations = []
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = fastdtw(
            self.locations, self.gt_locations, dist=self.euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance
            / (len(self.gt_locations) * SUCCESS_DISTANCE)
        )
        self._metric = nDTW

def run_interactive_user(episode_id, show_map):
    config = habitat.get_config("configs/tasks/vln_rxr.yaml")

    while True:
        config.defrost()

        config.DATASET.EPISODES_ALLOWED = [episode_id]

        config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
        config.TASK.AGENT_POSITION_SENSOR.TYPE = "agent_position"
        config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")

        config.TASK.AGENT_START_POSITION_SENSOR = habitat.Config()
        config.TASK.AGENT_START_POSITION_SENSOR.TYPE = "agent_start_position"
        config.TASK.SENSORS.append("AGENT_START_POSITION_SENSOR")

        config.TASK.NDTW = habitat.Config()
        config.TASK.NDTW.TYPE = "NDTW"
        config.TASK.MEASUREMENTS.append("NDTW")

        config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")

        config.freeze()

        env = SimpleRLEnv(
            config=config
        )

        print("Environment creation successful")
        observations = env.reset()
        print_scene_recur(env.habitat_env.sim.semantic_scene)
        print(f"Instruction text: [{observations['instruction']['text']}]\nAgent is at [{str(observations['agent_position'])}].\nAgent start position is at [{str(observations['agent_start_position'])}].")

        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

        if show_map.lower() == "y":
            ## debug map
            agent_position = env.habitat_env.sim.get_agent(0).state.position
            agent_heading = _quat_to_xy_heading(env.habitat_env.sim.get_agent(0).state.rotation)
            height = agent_position[1]
            top_down_map = maps.get_topdown_map(env.habitat_env.sim.pathfinder, height=height)
            top_down_map = maps.TOP_DOWN_MAP_COLORS[top_down_map]
            a_x, a_y = maps.to_grid(
                agent_position[2],
                agent_position[0],
                top_down_map.shape[0:2],
                sim=env.habitat_env.sim,
            )
            top_down_map_w_agent = maps.draw_agent(
                            image=np.copy(top_down_map),
                            agent_center_coord=(a_x, a_y),
                            agent_rotation=agent_heading,
                            agent_radius_px=min(top_down_map.shape[0:2]) // 64,
                        )
            
            top_down_map_cropped = top_down_map[a_x-map_size_floor:a_x+map_size_ceil, a_y-map_size_floor:a_y+map_size_ceil]
            top_down_map_cropped_rotated = ndimage.interpolation.rotate(top_down_map_cropped, (agent_heading * 180/np.pi), order=0, reshape=False)
            # np.unique(top_down_map) #-> 0,1
            ## end - debug map

            cv2.imshow("TOPDOWN_LARGE", top_down_map_w_agent)
            cv2.imshow("TOPDOWN_SMALL", top_down_map_cropped_rotated)
            cv2.waitKey(0)

        count_steps = 0
        while not env.get_done(observations):
            keystroke = cv2.waitKey(0)

            if keystroke == ord(FORWARD_KEY):
                action = HabitatSimActions.MOVE_FORWARD
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                action = HabitatSimActions.TURN_LEFT
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                action = HabitatSimActions.TURN_RIGHT
                print("action: RIGHT")
            elif keystroke == ord(FINISH):
                action = HabitatSimActions.STOP
                print("action: FINISH")
            else:
                print("INVALID KEY")
                continue

            observations, reward, done, info = env.step(action)
            #observations = env.step(action)
            count_steps += 1

            print(f"Instruction text: [{observations['instruction']['text']}]\nAgent is at [{str(observations['agent_position'])}].\nAgent start position is at [{str(observations['agent_start_position'])}].")

            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

            if show_map.lower() == "y":
                ## debug map
                agent_position = env.habitat_env.sim.get_agent(0).state.position
                agent_heading = _quat_to_xy_heading(env.habitat_env.sim.get_agent(0).state.rotation)
                height = agent_position[1]

                top_down_map = maps.get_topdown_map(env.habitat_env.sim.pathfinder, height=height)
                top_down_map = maps.TOP_DOWN_MAP_COLORS[top_down_map]
                a_x, a_y = maps.to_grid(
                    agent_position[2],
                    agent_position[0],
                    top_down_map.shape[0:2],
                    sim=env.habitat_env.sim,
                )
                top_down_map_w_agent = maps.draw_agent(
                                image=np.copy(top_down_map),
                                agent_center_coord=(a_x, a_y),
                                agent_rotation=agent_heading,
                                agent_radius_px=min(top_down_map.shape[0:2]) // 64,
                            )
                
                top_down_map_cropped = top_down_map[a_x-map_size_floor:a_x+map_size_ceil, a_y-map_size_floor:a_y+map_size_ceil]
                top_down_map_cropped_rotated = ndimage.interpolation.rotate(top_down_map_cropped, (agent_heading * 180/np.pi), order=0, reshape=False)
                # np.unique(top_down_map) #-> 0,1
                ## end - debug map

                cv2.imshow("TOPDOWN_LARGE", top_down_map_w_agent)
                cv2.imshow("TOPDOWN_SMALL", top_down_map_cropped_rotated)
                cv2.waitKey(0)

        dist_to_goal = env.get_info(observations)["distance_to_goal"]
        print(f"Episode finished after {count_steps} steps.\nDISTANCE_TO_GOAL={dist_to_goal};Success={env.get_info(observations)['success']};SPL={env.get_info(observations)['spl']};NDTW={env.get_info(observations)['ndtw']}.")

        if (
            action == HabitatSimActions.STOP
            and dist_to_goal < 3.0
        ):
            print("you successfully navigated to destination point")
        else:
            print("your navigation was unsuccessful")

        cv2.destroyWindow("RGB")
        if show_map.lower() == "y":
            cv2.destroyWindow("TOPDOWN_LARGE")
            cv2.destroyWindow("TOPDOWN_SMALL")
        cv2.waitKey(10)
        env.close()

        next_response = input('Do you want to continue with next episode? Type "Y" or "N": ')
        if next_response.lower() == "y":
            episode_id = input('Enter next Episode Id: ')
            continue
        else:
            break

def run_interactive_gt(episode_id):
    config = habitat.get_config("configs/tasks/vln_rxr.yaml")

    while True:
        config.defrost()

        config.DATASET.EPISODES_ALLOWED = [episode_id]

        config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
        config.TASK.AGENT_POSITION_SENSOR.TYPE = "agent_position"
        config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")

        config.TASK.AGENT_START_POSITION_SENSOR = habitat.Config()
        config.TASK.AGENT_START_POSITION_SENSOR.TYPE = "agent_start_position"
        config.TASK.SENSORS.append("AGENT_START_POSITION_SENSOR")

        config.TASK.NAV_EPISODE = habitat.Config()
        config.TASK.NAV_EPISODE.TYPE = "episode"
        config.TASK.SENSORS.append("NAV_EPISODE")

        config.TASK.NDTW = habitat.Config()
        config.TASK.NDTW.TYPE = "NDTW"
        config.TASK.MEASUREMENTS.append("NDTW")

        config.freeze()

        env = habitat.Env(
            config=config
        )

        print("Environment creation successful")
        observations = env.reset()
        print_scene_recur(env.sim.semantic_scene)
        episode_det = observations["episode"]
        print(f"Episode={episode_det.episode_id}.\nInstruction text: [{observations['instruction']['text']}]\nAgent is at [{str(observations['agent_position'])}].\nAgent start position is at [{str(observations['agent_start_position'])}].")

        actions = load_gt_actions(episode_det)
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        print("Agent stepping inside environment, following Oracle actions.")

        count_steps = 0
        for action in actions:
            if action == HabitatSimActions.MOVE_FORWARD:
                print("action: FORWARD")
            elif action == HabitatSimActions.TURN_LEFT:
                print("action: LEFT")
            elif action == HabitatSimActions.TURN_RIGHT:
                print("action: RIGHT")
            elif action == HabitatSimActions.STOP:
                print("action: FINISH")
            else:
                print("INVALID ACTION")
                continue

            observations = env.step(action)
            count_steps += 1
            keystroke = cv2.waitKey(0)

            print(f"Episode={episode_det.episode_id}.\nInstruction text: [{observations['instruction']['text']}]\nAgent is at [{str(observations['agent_position'])}].\nAgent start position is at [{str(observations['agent_start_position'])}].")

            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

            keystroke = cv2.waitKey(0)
            if keystroke == ord(FINISH):
                action = HabitatSimActions.STOP
                print("Interrupted.")
                break

        dist_to_goal = env.get_metrics()["distance_to_goal"]
        print(f"Episode finished after {count_steps} steps.\nDISTANCE_TO_GOAL={dist_to_goal};Success={env.get_metrics()['success']};SPL={env.get_metrics()['spl']};NDTW={env.get_metrics()['ndtw']}.")

        if (
            action == HabitatSimActions.STOP
            and dist_to_goal < 3.0
        ):
            print("you successfully navigated to destination point")
        else:
            print("your navigation was unsuccessful")
        
        cv2.destroyWindow("RGB")
        cv2.waitKey(10)
        env.close()

        next_response = input('Do you want to continue with next episode? Type "Y" or "N": ')
        if next_response.lower() == "y":
            episode_id = input('Enter next Episode Id: ')
            continue
        else:
            break

def run_interactive_predictions(episode_id):
    config = habitat.get_config("configs/tasks/vln_rxr.yaml")
    
    while True:
        config.defrost()

        config.DATASET.EPISODES_ALLOWED = [episode_id]

        config.TASK.AGENT_POSITION_SENSOR = habitat.Config()
        config.TASK.AGENT_POSITION_SENSOR.TYPE = "agent_position"
        config.TASK.SENSORS.append("AGENT_POSITION_SENSOR")

        config.TASK.AGENT_START_POSITION_SENSOR = habitat.Config()
        config.TASK.AGENT_START_POSITION_SENSOR.TYPE = "agent_start_position"
        config.TASK.SENSORS.append("AGENT_START_POSITION_SENSOR")

        config.TASK.NAV_EPISODE = habitat.Config()
        config.TASK.NAV_EPISODE.TYPE = "episode"
        config.TASK.SENSORS.append("NAV_EPISODE")

        config.TASK.NDTW = habitat.Config()
        config.TASK.NDTW.TYPE = "NDTW"
        config.TASK.MEASUREMENTS.append("NDTW")

        config.freeze()

        env = habitat.Env(
            config=config
        )
        print("Environment creation successful")
        observations = env.reset()
        episode_det = observations["episode"]
        print(f"Episode={episode_det.episode_id}.\nInstruction text: [{observations['instruction']['text']}]\nAgent is at [{str(observations['agent_position'])}].\nAgent start position is at [{str(observations['agent_start_position'])}].")

        actions = load_predicted_actions(episode_det)
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        print("Agent stepping inside environment, following Oracle actions.")

        count_steps = 0
        for action in actions:
            if action == HabitatSimActions.MOVE_FORWARD:
                print("action: FORWARD")
            elif action == HabitatSimActions.TURN_LEFT:
                print("action: LEFT")
            elif action == HabitatSimActions.TURN_RIGHT:
                print("action: RIGHT")
            elif action == HabitatSimActions.STOP:
                print("action: FINISH")
            else:
                print("INVALID ACTION")
                continue

            observations = env.step(action)
            count_steps += 1
            print(f"Episode={episode_det.episode_id}.\nInstruction text: [{observations['instruction']['text']}]\nAgent is at [{str(observations['agent_position'])}].\nAgent start position is at [{str(observations['agent_start_position'])}].")
            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

            keystroke = cv2.waitKey(0)
            if keystroke == ord(FINISH):
                action = HabitatSimActions.STOP
                print("Interrupted.")
                break

        dist_to_goal = env.get_metrics()["distance_to_goal"]
        print(f"Episode finished after {count_steps} steps.\nDISTANCE_TO_GOAL={dist_to_goal};Success={env.get_metrics()['success']};SPL={env.get_metrics()['spl']};NDTW={env.get_metrics()['ndtw']}.")

        if (
            action == HabitatSimActions.STOP
            and dist_to_goal < 3.0
        ):
            print("you successfully navigated to destination point")
        else:
            print("your navigation was unsuccessful")

        cv2.destroyWindow("RGB")
        cv2.waitKey(10)
        env.close()

        next_response = input('Do you want to continue with next episode? Type "Y" or "N": ')
        if next_response.lower() == "y":
            episode_id = input('Enter next Episode Id: ')
            continue
        else:
            break

def get_arguments():
    parser = argparse.ArgumentParser(description="Habitat API Demo")
    parser.add_argument(
        "--type",
        default="user",
        help="Type of interactive play (user, gt, or pred)",
    )
    parser.add_argument(
        "--episode_id",
        default="2373",
        help="Enter Episode Id",
    )
    parser.add_argument(
        "--show_map",
        default="N",
        help="Do you want to show top-down maps?(Y/N)",
    )

    args = parser.parse_args()
    return args

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
        # for region in level.regions:
        #     print(
        #         f"Region id:{region.id}, category:{region.category.name()},"
        #         f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
        #     )
        #     for obj in region.objects:
        #         print(
        #             f"Object id:{obj.id}, category:{obj.category.name()},"
        #             f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
        #         )
        #         count += 1
        #         if count >= limit_output:
        #             return

if __name__ == "__main__":
    
    args = get_arguments()

    if args.type == "user":
        run_interactive_user(args.episode_id, args.show_map)
    elif args.type == "gt":
        run_interactive_gt(args.episode_id)
    elif args.type == "pred":
        run_interactive_predictions(args.episode_id)
    