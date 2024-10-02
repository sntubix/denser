import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from rich.console import Console; CONSOLE = Console(width=120)
import torch


def invert_transformation(rot, t):
    t = np.matmul(-rot.T, t)
    inv_translation = np.concatenate([rot.T, t[:, None]], axis=1)
    return np.concatenate([inv_translation, np.array([[0.0, 0.0, 0.0, 1.0]])])

def global_pcd(all_points,all_colors,sequ_frames,transform_matrix,voxel_size):
    
    points3d = np.vstack(all_points)
    points3d_rgb = np.vstack(all_colors)
    
    # mask = points3d[:, 2] > 0
    # points3d = points3d[mask]
    # points3d_rgb = points3d_rgb[mask]

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    pcd.colors = o3d.utility.Vector3dVector(points3d_rgb)

    # Downsample the point cloud using a voxel size of your choice
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    print(len(downsampled_pcd.points))
    # Extract the downsampled points and colors
    downsampled_points = np.asarray(downsampled_pcd.points,dtype=np.float32)
    downsampled_colors = np.asarray(downsampled_pcd.colors)

    points3D = torch.from_numpy(downsampled_points)

    points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
    )
   
    points3D_rgb = torch.from_numpy((downsampled_colors * 255).astype(np.uint8))
     
    if len(points3D) > 0:      
        CONSOLE.log(f"Loaded {points3D.shape[0]} points from frame {sequ_frames[0]} to {sequ_frames[1]}")

    out = {
            "points3D_xyz": points3D[...,:3],
            "points3D_rgb": points3D_rgb,
        }
    return out
    

def lidar_to_world(lidar_frame_path,downsample_factor,tracking_calibration,poses_imu_w_tracking,normalized_index,scale_factor):
    
    pc_to_opengl= np.array([
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    point_cloud = np.fromfile(lidar_frame_path, '<f4')
    point_cloud = np.reshape(point_cloud, (-1, 4))    # x, y, z, r

    reflectance = point_cloud[:, 3]
    # Normalize reflectance to [0, 1]
    norm_reflectance = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
    
    # Use a colormap to convert normalized reflectance to RGB
    cmap = plt.get_cmap('viridis')

    point3d =  point_cloud[...,:3]   #x,y,z
    point3d_rgb = cmap(norm_reflectance)[...,:3] # r,g,b
    if downsample_factor < 1.0:
        num_points = point3d.shape[0]
        sample_size = int(num_points * downsample_factor)
        indices = np.random.choice(num_points, sample_size, replace=False)
        point3d = point3d[indices]
        point3d_rgb = point3d_rgb[indices]
    
    # Extract calibration data
    imu2velo = tracking_calibration["Tr_imu2velo"]
    velo2imu = invert_transformation(imu2velo[:3, :3], imu2velo[:3, 3])

    imu2world = poses_imu_w_tracking[normalized_index]
    points3d_homogeneous = np.concatenate([point3d, np.ones((point3d.shape[0], 1))], axis=1)
    
    points_imu = (points3d_homogeneous @ velo2imu.T)

    points_w = (points_imu @ imu2world.T)
    points3d_w = (pc_to_opengl @ points_w.T).T[...,:3]

    
    points3d_w*=scale_factor
    # Add offsets 
    offsets = np.array([-0.05, -0.122, 0.1]) 
    points_xyz_offset = points3d_w + offsets
    

    return points_xyz_offset,point3d_rgb
    
def transform_pcd_to_object_frame(points, transform):
    x,y,z,rot = transform
    o2w = np.eye(4)
    o2w[:3,:3] = rot
    o2w[:3,3] = [x, y, z]
    
    w2o = np.linalg.inv(o2w)

    pts_obj_frame = []
    for pt in points:
        h_p = np.array([pt[0],pt[1],pt[2],1])
        p = w2o @ h_p
        p  = p [:3] / p [3]
        pts_obj_frame.append(p) 
    return pts_obj_frame

def make_obj_pcd(velodyne_path,transform_matrix,scale_factor,sequ_frames,tracking_calibration,poses_imu_w_tracking,annotations,downsample_factor=1.0,overwrite=True):
    
    all_points = []
    all_colors = []

    geometry_list = []
    all_lidars_frames = sorted(os.listdir(velodyne_path))#[65:66]
    obj_pcd={}
    object_lidars_path = velodyne_path[:-13]+"object_lidars"
    assert os.path.exists(velodyne_path), f"The specified velodyne path does not exist: {velodyne_path}"
    assert os.path.exists(object_lidars_path), f"The specified object lidars path does not exist: {object_lidars_path}"
    for lidar_frame in all_lidars_frames:
        if lidar_frame.endswith('.bin'):
            index = int(lidar_frame.replace('.bin',''))
            if index in range(sequ_frames[0], sequ_frames[1] + 1):
                normalized_index = index - sequ_frames[0]
                geometry_list = []
                
                if normalized_index < len(annotations):
                    for annot in annotations[normalized_index]:
                        
                        obj_id = annot[4]
                        if obj_id == -1.0:
                            continue
                        
                        yaw = -annot[3]
                        obj_center = np.array([annot[0],annot[1],annot[2]])# x, y, z
                        obj_extent =  np.array([annot[7],annot[6],annot[5]])# length, width, height
                        obj_type = annot[-1]
                        obj_key = str(int(obj_id.item()))
                        #lidar to world to opengl to offset
                        if obj_key not in obj_pcd and obj_key != '-1':
                            obj_pcd[obj_key] = {
                                'xyz': [],
                                'rgb': [],
                            }
                        #lidar to world
                        lidar_frame_path = os.path.join(velodyne_path, lidar_frame)
                        save_path = object_lidars_path + f"/{obj_key}.ply"
                        if os.path.exists(save_path) and not overwrite:
                            continue
                        points3d, point3d_rgb = lidar_to_world(lidar_frame_path,downsample_factor*0.3,tracking_calibration,poses_imu_w_tracking,normalized_index,scale_factor)
                        all_points.append(points3d)
                        all_colors.append(point3d_rgb)
                    
                        
                    
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(points3d)
                        pcd.colors = o3d.utility.Vector3dVector(point3d_rgb)

                        # create obj box
                        R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, yaw, 0])
                        box = o3d.geometry.OrientedBoundingBox(obj_center, R, obj_extent)
                        box.color = np.random.rand(3)
                        
                        in_bbox = box.get_point_indices_within_bounding_box(pcd.points)
                        inliers_pcd = pcd.select_by_index(in_bbox,invert=False)
                        
                        inliers_pcd_obj = transform_pcd_to_object_frame(inliers_pcd.points,[obj_center[0], obj_center[1], obj_center[2], R])
                        
                        obj_pcd[obj_key]['xyz'].extend(inliers_pcd_obj)
                        obj_pcd[obj_key]['rgb'].extend(np.asarray(inliers_pcd.colors))

                        
                        pcd_transformed = o3d.geometry.PointCloud()
                        pcd_transformed.points = o3d.utility.Vector3dVector(inliers_pcd_obj)
                        # pcd_transformed.paint_uniform_color([0, 1, 0])
                        
                        # world coordinate frame
                        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0,0,0])
                        geometry_list.append(coordinate_frame)

                        obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=obj_center)
                        geometry_list.append(pcd)
                        geometry_list.append(obj_frame)
                        geometry_list.append(box)
                    
    # o3d.visualization.draw_geometries(geometry_list)
                # if len(geometry_list):
                #     set_view_point(geometry_list,f"/home/mahmud/Documents/report/lidarframes/{lidar_frame[:-4]}.png")
    count = 0 
    for ob_id, pcd in obj_pcd.items():
        # print(f"{ob_id}:{len(pcd['xyz'])}")
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(pcd['xyz']).astype(np.float32))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(pcd['rgb']).astype(np.float32))
        if len(point_cloud.points) > 0:
            o3d.io.write_point_cloud(str(object_lidars_path + f"/{ob_id}.ply"), point_cloud)
            CONSOLE.log(f"Saved object_{ob_id}'s {len(pcd['xyz'])} points in {object_lidars_path}/{ob_id}.ply")
            count+=len(pcd['xyz'])      

    CONSOLE.log(f"Total object points : {count} ")               
    # sparse = global_pcd(all_points,all_colors,sequ_frames,transform_matrix,downsample_factor*0.05)
    # return sparse
    


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

"""Utility functions to allow easy re-use of common operations across dataloaders"""
from pathlib import Path
from typing import List, Tuple, Union
import enum

import cv2
import numpy as np
import torch
from PIL import Image


class SemanticType(enum.IntEnum):
    DEFAULT = 0
    GROUND = 1
    SKY = 2

def get_image_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_mask = pil_mask.resize(newsize, resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("The mask image should have 1 channel")
    return mask_tensor


def get_semantics_and_mask_tensors_from_path(
    filepath: Path, mask_indices: Union[List, torch.Tensor], scale_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Utility function to read segmentation from the given filepath
    If no mask is required - use mask_indices = []
    """
    if isinstance(mask_indices, List):
        mask_indices = torch.tensor(mask_indices, dtype=torch.int64).view(1, 1, -1)
    pil_image = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.NEAREST)
    image = np.array(pil_image, dtype="int64")
    if len(image.shape) == 3:
        image = image[:, :, 0]
    # TODO(zz): fix magic number.
    semantics = np.zeros_like(image)
    semantics[(image == 7) + (image == 8) + (image == 13) + (image == 14) + (image == 23) + (image == 24)] = SemanticType.GROUND.value
    semantics[image == 27] = SemanticType.SKY.value

    semantics = torch.from_numpy(semantics).unsqueeze(-1)
    mask = torch.sum(semantics == mask_indices, dim=-1, keepdim=True) == 0
    return semantics, mask


def get_depth_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float = 1,
    interpolation: int = cv2.INTER_NEAREST,
    depth_type=None
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if filepath.suffix == ".npy":
        image = np.load(filepath) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    elif filepath.suffix == ".npz":
        # depth for omnidata
        image = np.load(filepath)['arr_0'] #* scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    elif depth_type == "2x8bit" or filepath.suffix == ".png":
        depth_img = cv2.imread(str(filepath.absolute()))
        image = depth_img[:,:,0] + (depth_img[:,:,1] * 256)
        image=image.astype(np.float64) * scale_factor * 0.01
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        image = image.astype(np.float64) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=interpolation)
    return torch.from_numpy(image).unsqueeze(-1)

