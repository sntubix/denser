# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation utils
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import torch
import yaml
import numpy as np 

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.rich_utils import CONSOLE
from copy import deepcopy



def adjust_theta(rot,rad=35):
    # Small rotation angle around Y-axis (for example, 5 degrees)
    theta = np.radians(rad)

    # Rotation matrix for Y-axis
    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    # Resulting matrix after rotation
    M_prime = np.dot(rot, Ry)

    return M_prime


def get_adjusted_z_vals(means, delta_z):
    # Adjust the z-values by subtracting delta_z
    adjusted_z = means[:, 2] - delta_z
    
    return adjusted_z

def get_delete_mask(means, box):
    # Size of the input means array
    num_means = means.shape[0]
    
    # Create a mask of False values for each element in means
    delete_mask = torch.zeros(num_means, dtype=torch.bool)
    return delete_mask
 
def get_mask(means,box):

    #scene 2
    # SIZE_A = np.array([1.1, 1.5, 0.7])
    # bbox_center = torch.tensor([-0.03, 0.12, 0]) 
    # new_box_rot= adjust_theta(box.rot)
    # box.rot = new_box_rot

     #scene 1
    # SIZE_A = np.array([1, 1.8, 1])
    # bbox_center = torch.tensor([0, 0.15, 0])  
    # bbox_size = box.size*SIZE_A / 2.0
    # new_box_rot= adjust_theta(box.rot,rad=-50)
    # box.rot = new_box_rot

    SIZE_A = np.array([0.6, 1, 1])
    bbox_center = torch.tensor([0, 0.05, 0.085])
    new_box_rot= adjust_theta(box.rot,rad=-20)
    # box.rot = new_box_rot


    bbox_size = box.size*SIZE_A / 2.0
    # Define the bounding box in PyTorch
    min_bounds = bbox_center - bbox_size
    max_bounds = bbox_center + bbox_size

    # Filter points within the bounding box 
    in_bbox_mask = (
        (means[:, 0] >= min_bounds[0]) & (means[:, 0] <= max_bounds[0]) &
        (means[:, 1] >= min_bounds[1]) & (means[:, 1] <= max_bounds[1]) &
        (means[:, 2] >= min_bounds[2]) & (means[:, 2] <= max_bounds[2])
    )
    return in_bbox_mask


def swap_edit(pipeline,loaded_state):


    new_state = deepcopy(loaded_state)

    ids = [key for key in pipeline.model.object_annos.objects_meta.keys()]
    params = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
    object_model_names = [pipeline.model.get_object_model_name(int(id)) for id in ids]
    


    # objs_to_filter = ['object_2','object_3','object_4']
    # for obj in objs_to_filter:
    #     object_3_box = pipeline.model.object_annos.objects_meta[obj.replace('object_','')]
    #     obj3_attr_list = [attr for attr in new_state['pipeline'].keys() if obj in attr]
    #     object_3_means = new_state['pipeline'][[atr for atr in obj3_attr_list if 'means' in atr ][0]]
        
    #     in_bbox_mask = get_mask(object_3_means,object_3_box)


    #     for param in params:
    #         gauss_param =  [atr for atr in obj3_attr_list if param in atr ][0]
    #         filtered_param = new_state['pipeline'][gauss_param][in_bbox_mask]
    #         new_state['pipeline'][gauss_param] = filtered_param


    objs_to_remove = ['object_3','object_4','object_8','object_9','object_10']
    
    # objs_to_remove = ['object_83']
    for obj in objs_to_remove:

        object_3_box = pipeline.model.object_annos.objects_meta[obj.replace('object_','')]
        obj3_attr_list = [attr for attr in new_state['pipeline'].keys() if obj in attr]
        object_3_means = new_state['pipeline'][[atr for atr in obj3_attr_list if 'means' in atr ][0]]
        delete_mask = get_delete_mask(object_3_means,object_3_box)


        for param in params:
            gauss_param =  [atr for atr in obj3_attr_list if param in atr ][0]
            filtered_param = new_state['pipeline'][gauss_param][delete_mask]
            new_state['pipeline'][gauss_param] = filtered_param


    # objs_to_replace = ['object_4','object_3']
    # for obj in objs_to_replace:
    #     for attr in [attr for attr in new_state['pipeline'].keys() if obj in attr]:
    #         attr2 = attr.replace(obj, 'object_6')
            
    #         if attr2 not in new_state['pipeline'].keys():
    #             continue
    #         new_state['pipeline'][attr] = new_state['pipeline'][attr2]

    
    # objs_to_replace = ['object_9','object_7']
    # for obj in objs_to_replace:
    #     for attr in [attr for attr in new_state['pipeline'].keys() if obj in attr]:
    #         attr2 = attr.replace(obj, 'object_10')
            
    #         if attr2 not in new_state['pipeline'].keys():
    #             continue
    #         new_state['pipeline'][attr] = new_state['pipeline'][attr2]

    
    # object_3_means = new_state['pipeline'][[atr for atr in obj3_attr_list if 'means' in atr ][0]]
    # object_3_colors = torch.sigmoid(new_state['pipeline'][[atr for atr in obj3_attr_list if 'features_dc' in atr ][0]])

    # object_5_means =  new_state['pipeline'][  [atr for atr  in [attr for attr in new_state['pipeline'].keys() if 'object_5' in attr]  if 'means' in atr ][0] ]

    #add opacities to bus gaussians,comment this shit out when done
    # model_name = 'object_8'
    # obj_attr_list = [attr for attr in new_state['pipeline'].keys() if model_name in attr]
    # bus_opacities  = new_state['pipeline'][[atr for atr in obj_attr_list if 'opacities' in atr ][0]] 
    # bus_opacities*=10000
    # new_state['pipeline']['_model.all_models.{model_name}.gauss_params.opacities'] = bus_opacities
    return new_state


def eval_load_checkpoint(config: TrainerConfig, pipeline: Pipeline) -> Tuple[Path, int]:
    ## TODO: ideally eventually want to get this to be the same as whatever is used to load train checkpoint too
    """Helper function to load checkpointed pipeline

    Args:
        config (DictConfig): Configuration of pipeline to load
        pipeline (Pipeline): Pipeline instance of which to load weights
    Returns:
        A tuple of the path to the loaded checkpoint and the step at which it was saved.
    """
    assert config.load_dir is not None

    config.load_dir = Path("/home/users/maali") / config.load_dir
    if config.load_step is None:
        CONSOLE.print("Loading latest checkpoint from load_dir")
        # NOTE: this is specific to the checkpoint name format
        if not os.path.exists(config.load_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No checkpoint directory found at {config.load_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the checkpoint exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(config.load_dir))[-1]
    else:
        load_step = config.load_step

    load_path = config.load_dir / f"step-{load_step:09d}.ckpt"

    assert load_path.exists(), f"Checkpoint {load_path} does not exist"
    loaded_state = torch.load(load_path, map_location="cpu")

    # new_state = swap_edit(pipeline,loaded_state)

    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])

    CONSOLE.print(f":white_check_mark: Done loading checkpoint from {load_path}")
    # del loaded_state
    return load_path, load_step,loaded_state


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        update_config_callback: Callback to update the config before loading the pipeline


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)

    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[config.method_name].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    # load checkpoints from wherever they were saved
    # TODO: expose the ability to choose an arbitrary checkpoint
    config.load_dir = config.get_checkpoint_dir()
    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()
    

    # load checkpointed information
    checkpoint_path, step,loaded_state = eval_load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path, step,loaded_state
