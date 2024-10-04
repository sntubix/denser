"""
Denser kitti datamanager.
"""
import random
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Literal, Optional, Tuple, Type, Union
from copy import deepcopy
from nerfstudio.cameras.cameras import Cameras, CameraType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig,FullImageDatamanager
from denser.data.datasets.ubix_dataset import UbixDataset


@dataclass
class UbixDataManagerConfig(FullImageDatamanagerConfig):
    """A semantic datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: UbixDataManager)



class UbixDataManager(FullImageDatamanager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing semantic data.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    def create_train_dataset(self) -> UbixDataset:
        # self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        return UbixDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            use_depth=self.config.dataparser.use_depth,
        )

    def create_eval_dataset(self) -> UbixDataset:

        return UbixDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            use_depth=self.config.dataparser.use_depth,
        )
    # @property
    # def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
    #     """
    #     Pretends to be the dataloader for evaluation, it returns a list of (camera, data) tuples.
    #     """
    #     if self.dataparser_config.split_setting == 'reconstruction':
    #         eval_indices = self.train_dataparser_outputs.metadata['eval_indices']
    #         image_indices = tuple(eval_indices)
    #         data = [deepcopy(self.cached_train[i]) for i in image_indices]
            
    #         _cameras = deepcopy(self.train_dataset.cameras).to(self.device)
    #         cameras = []
    #         for i in range(len(image_indices)):
    #             data[i]["image"] = data[i]["image"].to(self.device)  
    #         for i in image_indices:
    #             cameras.append(_cameras[i : i + 1])
    #         assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"        
    #         return list(zip(cameras, data))
        
    #     else:
    #         image_indices = [i for i in range(len(self.eval_dataset))]
    #         data = deepcopy(self.cached_eval)
    #         _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
    #         cameras = []
    #         for i in image_indices:
    #             data[i]["image"] = data[i]["image"].to(self.device)
    #             cameras.append(_cameras[i : i + 1])
    #         assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
    #         return list(zip(cameras, data))
        
    # def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
    #     """Returns the next evaluation batch

    #     Returns a Camera instead of raybundle
    #     """
    #     if self.dataparser_config.split_setting == 'reconstruction':
    #         eval_indices = self.train_dataparser_outputs.metadata['eval_indices'].tolist()
    #         image_idx = eval_indices.pop(random.randint(0, len(eval_indices) - 1))
    #         # Make sure to re-populate the unseen cameras list if we have exhausted it
    #         if len(eval_indices) == 0:
    #             eval_indices = self.train_dataparser_outputs.metadata['eval_indices'].tolist()
    #             image_idx = eval_indices.pop(random.randint(0, len(eval_indices) - 1))
    #         data = deepcopy(self.cached_eval[image_idx])
    #         data["image"] = data["image"].to(self.device)
    #         assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
    #         camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
    #         return camera, data
    #     else:

    #         image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
    #         # Make sure to re-populate the unseen cameras list if we have exhausted it
    #         if len(self.eval_unseen_cameras) == 0:
    #             self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
    #         data = deepcopy(self.cached_eval[image_idx])
    #         data["image"] = data["image"].to(self.device)
    #         assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
    #         camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
    #         return camera, data
