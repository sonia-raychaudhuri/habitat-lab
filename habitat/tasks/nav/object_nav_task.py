# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.visualizations import maps, fog_of_war
# import habitat_baselines.common.map_utils
import quaternion
from PIL import Image
import time

try:
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_name_id: Optional[int] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: Config,
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.GOAL_SPEC_MAX_VAL - 1
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:

        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )

@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "episodic_gps"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-agent_position[2], agent_position[0]], dtype=np.float32
            )
        else:
            return agent_position.astype(np.float32)
        
@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(Sensor):
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "episodic_compass"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING
    
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid
    
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        return self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

@registry.register_sensor(name="ObjectMapSensor")
class ObjectMapSensor(Sensor):
    r"""
        Map with Goals and Distractors marked
    Args:
        sim: reference to the simulator for calculating task observations.
        config: sensor config
    Attributes:
        
    """
    cls_uuid: str = "object_map"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        
        self.mapCache = {}
        self.cache_max_size = config.cache_max_size
        self.map_size = config.map_size
        self.meters_per_pixel = config.meters_per_pixel
        self.num_samples = config.num_samples
        self.nav_threshold = config.nav_threshold
        self.map_channels = config.MAP_CHANNELS
        self.draw_border = config.draw_border   #false
        self.with_sampling = config.with_sampling # true
        self.mask_map = config.mask_map
        self.visibility_dist = config.VISIBILITY_DIST
        self.fov = config.FOV
        self.object_ind_offset = config.object_ind_offset
        self.channel_num = 1
        self.object_padding = config.object_padding
        
        self.is_chal_21 = config.is_chal_21
        if self.is_chal_21:
            self.object_to_datset_mapping = {'chair': 0, 'table': 1, 'picture': 2, 'cabinet': 3, 'cushion': 4, 
                                         'sofa': 5, 'bed': 6, 'chest_of_drawers': 7, 'plant': 8, 'sink': 9, 
                                         'toilet': 10, 'stool': 11, 'towel': 12, 'tv_monitor': 13, 'shower': 14, 
                                         'bathtub': 15, 'counter': 16, 'fireplace': 17, 'gym_equipment': 18, 'seating': 19, 
                                         'clothes': 20}
        else:
            self.object_to_datset_mapping = {'chair': 0, 'bed': 1, 'plant': 2, 'toilet': 3, 'tv_monitor': 4, 'sofa': 5}
        
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=10,
            shape=(self.map_size, self.map_size, self.map_channels),
            dtype=np.int,
        )
        
    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip
        
    def get_topdown_map(self,
        pathfinder,
        height: float,
        map_resolution: int = 1024,
        draw_border: bool = True,
        meters_per_pixel: Optional[float] = None,
        with_sampling: Optional[bool] = True,
        num_samples: Optional[float] = 50,
        nav_threshold: Optional[float] = 0.3,
    ) -> np.ndarray:
        r"""Return a top-down occupancy map for a sim. Note, this only returns valid
        values for whatever floor the agent is currently on.

        :param pathfinder: A habitat-sim pathfinder instances to get the map from
        :param height: The height in the environment to make the topdown map
        :param map_resolution: Length of the longest side of the map.  Used to calculate :p:`meters_per_pixel`
        :param draw_border: Whether or not to draw a border
        :param meters_per_pixel: Overrides map_resolution an

        :return: Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
            the flag is set).
        """

        if meters_per_pixel is None:
            meters_per_pixel = maps.calculate_meters_per_pixel(
                map_resolution, pathfinder=pathfinder
            )

        if with_sampling:
            top_down_map = pathfinder.get_topdown_view_with_sampling(
                meters_per_pixel=meters_per_pixel, height=height,
                num_samples=num_samples, nav_threshold=nav_threshold
            ).astype(np.uint8)
        else:
            top_down_map = pathfinder.get_topdown_view(
                meters_per_pixel=meters_per_pixel, height=height
            ).astype(np.uint8)

        # Draw border if necessary
        if draw_border:
            maps._outline_border(top_down_map)

        return np.ascontiguousarray(top_down_map)


    def get_topdown_map_from_sim(self,
        sim: "HabitatSim",
        map_resolution: int = 1024,
        draw_border: bool = True,
        meters_per_pixel: Optional[float] = None,
        agent_id: int = 0,
        with_sampling: Optional[bool] = True,
        num_samples: Optional[float] = 50,
        nav_threshold: Optional[float] = 0.3,
    ) -> np.ndarray:
        r"""Wrapper around :py:`get_topdown_map` that retrieves that pathfinder and heigh from the current simulator

        :param sim: Simulator instance.
        :param agent_id: The agent ID
        """
        return self.get_topdown_map(
            sim.pathfinder,
            sim.get_agent(agent_id).state.position[1],
            map_resolution,
            draw_border,
            meters_per_pixel,
            with_sampling,
            num_samples,
            nav_threshold
        )

    def print_scene_recur(self, scene, limit_output=10):
        print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
        print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

        # count = 0
        # for level in scene.levels:
        #     print(
        #         f"Level id:{level.id}, center:{level.aabb.center},"
        #         f" dims:{level.aabb.sizes}"
        #     )
        #     count += 1
        #     if count >= limit_output:
        #         break

        count = 0
        for obj in scene.objects:
            print(
                f"Object id:{obj.id}, category:{obj.category.name()},"
                f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            )
            count += 1
            # if count >= limit_output:
            #     break
                    
        # count = 0
        # for region in scene.regions:
        #     print(
        #         f"Region id:{region.id}, category:{region.category.name()},"
        #         f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
        #     )
        #     count += 1
        #     if count >= limit_output:
        #         break
                    
    def match_sensor_cat(self, scene, observations, episode):
        
        sem_observed = np.unique(observations['semantic'])
        scene_object_ids = [o.id for o in scene.objects]
        
        if set(sem_observed).issubset(set(scene_object_ids)):
            for s in sem_observed:
                print(scene.objects[scene.objects==s])
            
    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        # debug
        scene = self._sim.semantic_scene
        # self.print_scene_recur(scene)
        self.match_sensor_cat(scene, observations, episode)
        #
        
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        agent_vertical_pos = str(round(agent_position[1], 2))
        
        if (episode.scene_id in self.mapCache and 
                agent_vertical_pos in self.mapCache[episode.scene_id]):
            top_down_map = self.mapCache[episode.scene_id][agent_vertical_pos].copy()
            
        else:
            top_down_map = self.get_topdown_map_from_sim(
                self._sim,
                draw_border=self.draw_border,
                meters_per_pixel=self.meters_per_pixel,
                with_sampling=False, #self.with_sampling,
                num_samples=self.num_samples,
                nav_threshold=self.nav_threshold
            )
            if episode.scene_id not in self.mapCache:
                if len(self.mapCache) > self.cache_max_size:
                    # Reset cache when cache size exceeds max size
                    self.mapCache = {}
                self.mapCache[episode.scene_id] = {}
            self.mapCache[episode.scene_id][agent_vertical_pos] = top_down_map.copy()
            
        object_map = np.zeros((top_down_map.shape[0], top_down_map.shape[1], self.map_channels))
        object_map[:top_down_map.shape[0], :top_down_map.shape[1], 0] = top_down_map

        # Get agent location on map
        agent_loc = maps.to_grid(
                    agent_position[2],
                    agent_position[0],
                    top_down_map.shape[0:2],
                    sim=self._sim,
                )

        # Mark the agent location
        object_map[max(0, agent_loc[0]-self.object_padding):min(top_down_map.shape[0], agent_loc[0]+self.object_padding),
                    max(0, agent_loc[1]-self.object_padding):min(top_down_map.shape[1], agent_loc[1]+self.object_padding),
                    self.channel_num+1] = 10 #len(self.object_to_datset_mapping) + self.object_ind_offset
        

        # Mask the map
        if self.mask_map:
            _fog_of_war_mask = np.zeros_like(top_down_map)
            _fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                top_down_map,
                _fog_of_war_mask,
                np.array(agent_loc),
                self.get_polar_angle(),
                fov=self.fov,
                max_line_len=self.visibility_dist
                / self.meters_per_pixel,
            )
            object_map[:, :, self.channel_num] += 1
            object_map[:, :, self.channel_num] *= _fog_of_war_mask # Hide unobserved areas

        # Adding goal information on the map
        for i in range(len(episode.goals)):
            loc0 = episode.goals[i].position[0]
            loc2 = episode.goals[i].position[2]
            grid_loc = maps.to_grid(
                loc2,
                loc0,
                top_down_map.shape[0:2],
                sim=self._sim,
            )
            object_map[grid_loc[0]-self.object_padding:grid_loc[0]+self.object_padding, 
                        grid_loc[1]-self.object_padding:grid_loc[1]+self.object_padding,
                        self.channel_num] = (
                                self.object_to_datset_mapping[episode.goals[i].object_category]
                                + self.object_ind_offset
                            )

        # Hide the  out-of-view objects
        if self.mask_map:
            object_map[:, :, self.channel_num] *= _fog_of_war_mask   
            
        final_object_map = np.zeros((self.map_size, self.map_size, self.map_channels))
        final_object_map[:top_down_map.shape[0], :top_down_map.shape[1], :] = object_map

        return final_object_map

@registry.register_sensor(name="ObjectNavSemanticSensor")
class ObjectNavSemanticSensor(Sensor):
    r"""
        Map with ground-truth object nav goal category
    Args:
        sim: reference to the simulator for calculating task observations.
        config: sensor config
    Attributes:
        
    """
    cls_uuid: str = "semantic_labels"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self.height = 256
        self.width = 256
        self.chal_version = 'v2' #'v1'
        
        if self.chal_version == 'v1':
            self.object_to_datset_mapping = {'chair': 1, 'table': 2, 'picture': 3, 'cabinet': 4, 'cushion': 5, 
                                            'sofa': 6, 'bed': 7, 'chest_of_drawers': 8, 'plant': 9, 'sink': 10, 
                                            'toilet': 11, 'stool': 12, 'towel': 13, 'tv_monitor': 14, 'shower': 15, 
                                            'bathtub': 16, 'counter': 17, 'fireplace': 18, 'gym_equipment': 19, 'seating': 20, 
                                            'clothes': 21}
            self.category_to_mp3d_category_id = {'chair': 3, 'table': 5, 'picture': 6, 'cabinet': 7, 'cushion': 8,
                                                 'sofa': 10, 'bed': 11, 'chest_of_drawers': 13, 'plant': 14, 'sink': 15,
                                                 'toilet': 18, 'stool': 19, 'towel': 20, 'tv_monitor': 22, 
                                                 'shower': 23, 'bathtub': 25, 'counter': 26, 'fireplace': 27, 
                                                 'gym_equipment': 33, 'seating': 34, 'clothes': 38}
            self.category_to_task_category_id = {'chair': 0, 'table': 1, 'picture': 2, 'cabinet': 3, 'cushion': 4, 
                                                 'sofa': 5, 'bed': 6, 'chest_of_drawers': 7, 'plant': 8, 'sink': 9, 
                                                 'toilet': 10, 'stool': 11, 'towel': 12, 'tv_monitor': 13, 
                                                 'shower': 14, 'bathtub': 15, 'counter': 16, 'fireplace': 17, 
                                                 'gym_equipment': 18, 'seating': 19, 'clothes': 20}
        else:
            self.object_to_datset_mapping = {'chair': 1, 'bed': 2, 'plant': 3, 
                                         'toilet': 4, 'tv_monitor': 5, 'sofa': 6,
                                         'tv': 5, 'monitor': 5, 'couch': 6}
            
        
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.iinfo(np.uint32).min,
            high=np.iinfo(np.uint32).max,
            shape=(self.height, self.width, 1),
            dtype=np.int32,
        )
            
    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        scene = self._sim.semantic_scene
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        instance_id_to_label_names = {int(obj.id.split("_")[-1]): obj.category.name() for obj in scene.objects}
        
        semantic_obs = np.array(observations['semantic']).astype(np.uint8).squeeze(-1)
        
        # instance id to MP3D category id
        # semantic_obs_transformed = np.array(
        #                             [instance_id_to_label_id[s] for s in semantic_obs.reshape(-1)]
        #                         ).reshape(semantic_obs.shape)
        
        # instance id to MP3D category to ObjNav category id
        semantic_transformed = np.array(
                                    [self.object_to_datset_mapping[instance_id_to_label_names[s]]
                                        if instance_id_to_label_names[s] in self.object_to_datset_mapping
                                        else 0
                                     for s in semantic_obs.reshape(-1)]
                                ).reshape(semantic_obs.shape).astype(np.uint8)
        
        ## debug
        # img_folder = '/localhome/sraychau/Projects/Research/ObjNav/habitat-lab/test_images/mp3d'
        # file_id = str(time.time())
        # rgb_obs = np.array(observations['rgb'])
        # Image.fromarray(rgb_obs).save(os.path.join(img_folder, f'{file_id}_rgb.jpg'))
        # Image.fromarray(maps.TOP_DOWN_MAP_COLORS[semantic_obs]).save(os.path.join(img_folder, f'{file_id}_sem.jpg'))
        # Image.fromarray(maps.TOP_DOWN_MAP_COLORS[semantic_transformed]).save(os.path.join(img_folder, f'{file_id}_sem_transformed.jpg'))

        return semantic_transformed

@registry.register_sensor(name="PositionSensor")
class AgentPositionSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return self._sim.get_agent_state().position

@registry.register_sensor(name="RotationSensor")
class RotationSensor(Sensor):
    r"""The agent's world rotation as quaternion
        -   similar to CompassSensor
    """
    cls_uuid: str = "agent_rotation"
    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(34,),
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        return quaternion.as_float_array(rotation_world_agent)

@registry.register_task(name="ObjectNav-v1")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
