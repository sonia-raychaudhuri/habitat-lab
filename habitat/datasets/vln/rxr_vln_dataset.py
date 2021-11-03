import gzip
import json
import os
from typing import Dict, List, Optional, Union
import time

import numpy as np
import attr
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import ALL_SCENES_MASK
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
SLURM_TMPDIR = os.environ.get("SLURM_TMPDIR", None)

def is_slurm_job() -> bool:
    return SLURM_JOBID is not None

@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(
        default=None
    )
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)
    instruction_features: Optional[List[float]] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        r"""Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls._scene_from_episode(episode) for episode in dataset.episodes}
        )

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._scene_from_episode(episode) in scenes_to_load
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            if episode["scene_id"] not in ["mp3d/VFuaQ6m2Qom/VFuaQ6m2Qom.glb"]:
                continue
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNCEDatasetV1(Dataset):
    r"""Loads the RxR VLN-CE Dataset."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        r"""Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> str:
        return episode.instruction.language

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls._scene_from_episode(episode) for episode in dataset.episodes}
        )

    @classmethod
    def extract_roles_from_config(cls, config: Config) -> List[str]:
        if ALL_ROLES_MASK in config.ROLES:
            return cls.annotation_roles
        assert set(config.ROLES).issubset(set(cls.annotation_roles))
        return config.ROLES

    @classmethod
    def check_config_paths_exist(cls, config: Config) -> bool:
        if is_slurm_job() and SLURM_TMPDIR is not None:
            scenes_dir = os.path.join(SLURM_TMPDIR, config.SCENES_DIR)
        else:
            scenes_dir = config.SCENES_DIR
        return all(
            os.path.exists(
                config.DATA_PATH.format(split=config.SPLIT, role=role)
            )
            for role in cls.extract_roles_from_config(config)
        ) and os.path.exists(scenes_dir)

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        if is_slurm_job() and SLURM_TMPDIR is not None:
            scenes_dir = os.path.join(SLURM_TMPDIR, config.SCENES_DIR)
        else:
            scenes_dir = config.SCENES_DIR

        for role in self.extract_roles_from_config(config):
            with gzip.open(
                config.DATA_PATH.format(split=config.SPLIT, role=role), "rt"
            ) as f:
                self.from_json(f.read(), scenes_dir=scenes_dir)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._scene_from_episode(episode) in scenes_to_load
            ]

        if ALL_LANGUAGES_MASK not in config.LANGUAGES:
            languages_to_load = set(config.LANGUAGES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]

        if config.EPISODES_ALLOWED is not None:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def _load_instruction_features(self, episode: VLNEpisode):
        self.features_path = self.config.INSTRUCTION_FEATURES_PATH
        features = np.load(
            self.features_path.format(
                split=episode.instruction.split,
                id=int(episode.instruction.instruction_id),
                lang=episode.instruction.language.split("-")[0],
            )
        )
        feats = np.zeros((512, 768), dtype=np.float32)
        s = features["features"].shape
        feats[: s[0], : s[1]] = features["features"]
        return feats

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)

        _loading_start = time.time()
        for episode in deserialized["episodes"]:
            if self.config.EPISODES_ALLOWED is not None and episode["episode_id"] not in self.config.EPISODES_ALLOWED:
                continue
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )
            episode.instruction.split = self.config.SPLIT
            #episode.instruction.instruction_features = self._load_instruction_features(episode)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

