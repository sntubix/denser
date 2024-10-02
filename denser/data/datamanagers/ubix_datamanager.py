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
from denser.data.datasets.ubix_dataset import UbixDataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig,FullImageDatamanager
from nerfstudio.cameras.cameras import Cameras, CameraType



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


    
    
   