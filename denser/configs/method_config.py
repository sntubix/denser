from __future__ import annotations

from pathlib import Path
from typing import Dict

import tyro

# from denser.data.datamanagers.ubix_datamanager import FullImageDatamanagerConfig,
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
import torch
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.plugins.types import MethodSpecification

from denser.data.datamanagers.ubix_datamanager import UbixDataManagerConfig, UbixDataManager 
from denser.data.dataparsers.ubix_kitti_dataparser import MarsKittiDataParserConfig
from denser.models.scene_model import SceneModelConfig 
from denser.models.splatfacto import SplatfactoModelConfig


denser =  MethodSpecification(

    config = TrainerConfig(
        method_name="denser",
        steps_per_eval_image=1000,
        steps_per_eval_batch=1000,
        steps_per_save=2000,
        steps_per_eval_all_images=5000,
        max_num_iterations=200000,
        mixed_precision=False,

        pipeline = VanillaPipelineConfig(
            datamanager= UbixDataManagerConfig( 
                dataparser = MarsKittiDataParserConfig(use_depth=True,
                                                       scale_factor=0.1,
                                                       split_setting= 'reconstruction'),
                cache_images_type="uint8",
                ),

            model= SceneModelConfig(
                    background_model = SplatfactoModelConfig(
                        cull_alpha_thresh=0.002,
                        cull_scale_thresh=0.2,
                        densify_grad_thresh=0.0002,
                        warmup_length=500,
                        refine_every=100,
                        reset_alpha_every=30,
                        wavelets_features_dim=1,
                        stop_split_at=25000,
                ),

                object_model_template=SplatfactoModelConfig(
                        cull_alpha_thresh=0.002,
                        cull_scale_thresh=0.2,
                        densify_grad_thresh=0.0002,
                        warmup_length=500,
                        refine_every=100,
                        stop_split_at=25000,
                        wavelets_features_dim=16,
                        num_random=10000,
                        
                )
            
            ),
        ),
        optimizers={
           
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
           
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "scale_param": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),"scheduler": None,
            },
            "translation_param": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),"scheduler": None,
            },
        },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer+wandb",
    ),
    description="Gaussian Scene model implementation with gaussian splatting model for backgruond and object models.",

)