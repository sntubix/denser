
import os 
import numpy as np
import open3d as o3d
import torch
import bisect
from copy import deepcopy

from nerfstudio.cameras.camera_utils import quaternion_from_matrix, quaternion_slerp, quaternion_matrix
from nerfstudio.utils.rich_utils import CONSOLE

EXP_RATE = np.array([1.3, 1.3, 1.1])

SIZE_A = np.array([0, 0, 0.001])

def create_straight_trajectory_with_sharp_orientation_change(trajectory_map, object_orientation_map, car_id):
    """
    Modifies the trajectory of a selected car to follow a straight path in the initial direction,
    with sharper orientation changes towards the end of the trajectory.

    Parameters:
    - trajectory_map: Dictionary containing car IDs as keys and lists of 3D positions as values.
    - object_orientation_map: Dictionary containing car IDs as keys and lists of 3D orientations as values.
    - car_id: The ID of the car to modify.

    Returns:
    - Updated trajectory and orientation maps with the new straight path and sharp orientation transition.
    """
    # Check if the car_id exists in the trajectory map
    if car_id not in trajectory_map:
        print(f"Car ID {car_id} not found in trajectory map.")
        return trajectory_map, object_orientation_map

    # Extract the original trajectory and orientation for the given car ID
    original_trajectory = np.array(trajectory_map[car_id])
    original_orientation = np.array(object_orientation_map[car_id])

    # Get initial position and direction
    start_point = original_trajectory[0]
    initial_orientation = original_orientation[0]
    
    # Calculate the initial forward direction (assuming Z-axis is forward)
    forward_direction = -initial_orientation[:, 2]  # Assuming third column is the forward vector

    # Normalize the forward direction
    if np.linalg.norm(forward_direction) != 0:
        forward_direction = forward_direction / np.linalg.norm(forward_direction)

    # Create a new trajectory that goes straight
    new_trajectory = np.zeros_like(original_trajectory)
    for i in range(len(new_trajectory)):
        new_trajectory[i] = start_point + forward_direction * np.linalg.norm(original_trajectory[i] - start_point)

    # Smoothly and sharply adjust orientations to gradually straighten the path
    new_orientation = np.zeros_like(original_orientation)
    for i in range(len(new_orientation)):
        # Calculate a non-linear interpolation factor for sharper change towards the end
        t = (i / (len(new_orientation) - 1))**6  # Quadratic interpolation for sharper change

        # Interpolate between the current orientation and the initial straight orientation
        current_orientation = original_orientation[i]
        new_orientation[i] = (1 - t) * current_orientation + t * initial_orientation

        # Normalize the new orientation to maintain orthogonality
        u, _, vh = np.linalg.svd(new_orientation[i])  # SVD for orthogonalization
        new_orientation[i] = np.dot(u, vh)

    # Update the trajectory and orientation maps with the new data
    trajectory_map[car_id] = new_trajectory.tolist()
    object_orientation_map[car_id] = new_orientation.tolist()

    return trajectory_map, object_orientation_map

def create_straight_trajectory_from_initial(trajectory_map, object_orientation_map, car_id):
    """
    Modifies the trajectory of a selected car to follow a straight path in the initial direction,
    and sets all orientations to match the initial orientation.

    Parameters:
    - trajectory_map: Dictionary containing car IDs as keys and lists of 3D positions as values.
    - object_orientation_map: Dictionary containing car IDs as keys and lists of 3D orientations as values.
    - car_id: The ID of the car to modify.

    Returns:
    - Updated trajectory and orientation maps with the new straight path and uniform orientation.
    """
    # Check if the car_id exists in the trajectory map
    if car_id not in trajectory_map:
        print(f"Car ID {car_id} not found in trajectory map.")
        return trajectory_map, object_orientation_map

    # Extract the original trajectory and orientation for the given car ID
    original_trajectory = np.array(trajectory_map[car_id])
    original_orientation = np.array(object_orientation_map[car_id])

    # Get initial position and direction
    start_point = original_trajectory[0]
    initial_orientation = original_orientation[16]
    
    # Calculate the initial forward direction (assuming Z-axis is forward)
    forward_direction = -initial_orientation[:, 2]  # Assuming third column is the forward vector

    # Normalize the forward direction
    if np.linalg.norm(forward_direction) != 0:
        forward_direction = forward_direction / np.linalg.norm(forward_direction)

    # Create a new trajectory that goes straight
    new_trajectory = np.zeros_like(original_trajectory)
    for i in range(len(new_trajectory)):
        new_trajectory[i] = start_point + forward_direction * np.linalg.norm(original_trajectory[i] - start_point)

    # Set all orientations to the initial orientation
    new_orientation = np.array([initial_orientation for _ in range(len(new_trajectory))])

    # Update the trajectory and orientation maps with the new data
    trajectory_map[car_id] = new_trajectory.tolist()
    object_orientation_map[car_id] = new_orientation.tolist()

    return trajectory_map, object_orientation_map

def create_left_turn_trajectory(trajectory_map, object_orientation_map, car_id, turn_strength=0.1, orientation_change=0.01):
    """
    Modifies the trajectory of a selected car to create a left-turn path,
    ensuring the same change is applied to repeated poses and slightly adjusting the orientation.

    Parameters:
    - trajectory_map: Dictionary containing car IDs as keys and lists of 3D positions as values.
    - object_orientation_map: Dictionary containing car IDs as keys and lists of 3D orientations as values.
    - car_id: The ID of the car to modify.
    - turn_strength: The strength of the turn (how sharp the left turn is).
    - orientation_change: The amount of orientation change applied to the car's orientation matrices.

    Returns:
    - Updated trajectory and orientation maps with the new left-turn path and orientation.
    """
    # Check if the car_id exists in the trajectory map
    if car_id not in trajectory_map:
        print(f"Car ID {car_id} not found in trajectory map.")
        return trajectory_map, object_orientation_map

    # Extract the original trajectory and orientation for the given car ID
    new_trajectory = np.array(trajectory_map[car_id])
    new_orientation = np.array(object_orientation_map[car_id])

    # Create a new trajectory that introduces a left turn
    num_points = len(new_trajectory)

    # Track which poses have been modified to apply the same transformation to duplicates
    modified_poses = {}
    modified_orientations = {}

    # Define a small rotation matrix for orientation change
    rotation_angle = orientation_change  # Small angle for incremental rotation
    rotation_matrix = np.array([
        [np.cos(rotation_angle), 0, np.sin(rotation_angle)],
        [0, 1, 0],  # Y-axis remains unchanged for slight left turn
        [-np.sin(rotation_angle), 0, np.cos(rotation_angle)]
    ])

    # Apply a gradual left turn to the trajectory and adjust orientation
    for i in range(num_points):
        # Convert pose to a tuple to use as a key in the dictionary
        pose_key = tuple(new_trajectory[i])
        
        # Check if this pose has been modified before
        if pose_key in modified_poses:
            # Use the same transformation for repeated poses
            new_trajectory[i] = modified_poses[pose_key]
            new_orientation[i] = modified_orientations[pose_key]
        else:
            # Apply transformation to introduce a left turn
            new_trajectory[i, 0] += turn_strength * i / num_points  # Adjust x-coordinate
            new_trajectory[i, 2] += turn_strength * np.sin(i / num_points * np.pi)  # Adjust z-coordinate
            
            # Apply rotation to the orientation matrix
            new_orientation[i] = new_orientation[i] @ rotation_matrix
            
            # Store the modified pose and orientation
            modified_poses[pose_key] = new_trajectory[i]
            modified_orientations[pose_key] = new_orientation[i]

    # Update the trajectory and orientation maps with the new data
    trajectory_map[car_id] = new_trajectory.tolist()
    object_orientation_map[car_id] = new_orientation.tolist()

    return trajectory_map, object_orientation_map

def make_new_trafjectory(annots,car_id,turn_strength=-4.7, orientation_change=0.2,trj='left'):

    annotations = deepcopy(annots)
    object_trajectory_map = {}
    object_orientation_map = {}

    for index in range(len(annotations)):
        boxes_in_frame = annotations[str(index)]
        for box in boxes_in_frame:
            trackId = str(box.trackId)
            if trackId not in object_trajectory_map:
                object_trajectory_map[trackId] = [box.center]
                object_orientation_map[trackId] = [box.rot]
            else:
                object_trajectory_map[trackId].append(box.center)
                object_orientation_map[trackId].append(box.rot)
    
    # Make deep copies of the trajectory and orientation maps
    new_trajectory_map = deepcopy(object_trajectory_map)
    new_orientation_map = deepcopy(object_orientation_map)

    if trj == 'left':    
        new_trajectory_map, new_orientation_map = create_left_turn_trajectory(new_trajectory_map, new_orientation_map,car_id, turn_strength=turn_strength, orientation_change=orientation_change)
    else:
        new_trajectory_map, new_orientation_map = create_straight_trajectory_with_sharp_orientation_change(new_trajectory_map, new_orientation_map,car_id)
    
    map_index = 0
    for frame in range(len(annotations)):
        boxes_in_frame = annotations[str(frame)]
        for box in boxes_in_frame:
            trackId = str(box.trackId)
            
            if trackId != car_id:
                continue
            elif trackId == car_id :
                new_pose = new_trajectory_map[trackId][map_index]
                new_orientation = new_orientation_map[trackId][map_index]
                box.center = new_pose
                box.rot = new_orientation
                map_index+=1

        
    return object_trajectory_map, object_orientation_map, new_trajectory_map, new_orientation_map,annotations

def frame_interpolation(frame_i1, frame_i2, frame_id):
    # get intersection  box
    trackId_i1 = {i.trackId: i for i in frame_i1}
    trackId_i2 = {i.trackId: i for i in frame_i2}
    intersection_trackId = list(
        set(trackId_i1.keys()) & set(trackId_i2.keys()))
    i_frame = []
    for trackId in intersection_trackId:
        box_i1 = trackId_i1[trackId]
        box_i2 = trackId_i2[trackId]
        iterpolation_box = BoundingBox.interploate(box_i1, box_i2, frame_id=frame_id)
        i_frame.append(iterpolation_box)
    return i_frame

class BoundingBox:
    def __init__(self,center, yaw=None, trackId=None, size=None, label=None, frame_id=-1, frame=-1, rot=None, quat=None):
        self.trackId = trackId
        self.center = center
        
        self.croped_pc = None
        # self.yaw=yaw
        if rot is not None:
            assert rot.shape == (3, 3)
            self.rot = rot
        else:
            self.rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, yaw, 0])
     
        self.size = size
        self.label = label
        self.frame_id = int(frame_id)
        self.frame = int(frame)
        self.quat = quat

    def transform(self, translation, rotation):
        self.center = np.dot(rotation, self.center) #+translation
        self.rot = np.dot(rotation, self.rot)
        

class Annotations:
    
    def __init__(self, annotations, lidar_path=None, transform_matrix: np.ndarray = None) -> None:
            
        self.annotations = []
        if lidar_path is not None:
            assert os.path.exists(lidar_path), f"Lidar path {lidar_path} does not exist"

        if annotations is not None:
            self.annotations = annotations
        self.lidar_path = lidar_path
        
        self.transform_matrix = np.eye(4) if transform_matrix is None else transform_matrix

        # mapping timestamp to list of Box object
        self.annos = {}
        # mapping trackID to each object's seed_pts
        self.seed_pts = {}
        
        # mapping trackID to Box object
        self.objects_meta = {}
       
        self.objects_frames = {}
       
        for index, items in enumerate(self.annotations):
            
            self.annos[str(index)] = self.make_box_obj(index,items)

        # _, _, _, _,new_annotations = make_new_trafjectory(self.annos,'8',turn_strength=-4.7, orientation_change=0.2,trj='straight')
        # self.annos = new_annotations

        self.all_names = list(self.annos.keys())
        self.unique_track_ids = list(self.objects_meta.keys())

    def __len__(self):
        return len(self.all_names)

    def get_by_id(self, index):
        return self.annos[self.all_names[index]]
    
    def get_seed_pts(self, obj_id):
        return self.seed_pts[obj_id]
    

    def update(self, frame_id, boxes):
        frame_id = str(int(frame_id))
        self.all_names.append(frame_id)
        self.all_names.sort()
        self.annos.update({frame_id: boxes})
        self.annos = dict(sorted(self.annos.items(), key=lambda x: int(x[0])))
    
    def  __getitem__(self,frame_id):
        
        if not len(self):
            return []
        # frame_id = str(frame_id)
        
        if isinstance(frame_id, (int, float)):
            # check within 0-1, assume it is a portion of the whole sequence rather than a timestamp
            if isinstance(frame_id,float) and frame_id>=0 and frame_id<=1 and len(self.all_names):
                frame_id = min(round(frame_id*len(self.all_names)),len(self.all_names)-1)
                frame_id = self.all_names[frame_id]
                print('@'*50,'frame_id is a portion of the whole sequence, use frame_id',frame_id)
            else:
                frame_id =  str(int(frame_id))
        elif isinstance(frame_id, str):
            frame_id = frame_id
        else:
            raise ValueError('frame_id should be int or str')
        
        if frame_id in self.all_names:
            return self.annos[frame_id]
        # find nearest frame_id
        if frame_id < self.all_names[0] or frame_id > self.all_names[-1]:
            # print('@'*50,'frame_id',frame_id,'out of range')
            return []

        nearest_frame_id = bisect.bisect(
            [int(i) for i in self.all_names], int(frame_id))

        # print('@'*50,'nearest_frame_id-1',nearest_frame_id-1,'nearest_frame_id',nearest_frame_id,'frame_id',frame_id,self.all_names[nearest_frame_id-1])
        frame_i1, frame_i2 = self.all_names[nearest_frame_id -
                                            1], self.all_names[nearest_frame_id]
        # print('@'*50,"use interpolation, query frame_id",frame_id,'nearest_frame_id',nearest_frame_id,'frame_i1',frame_i1,'frame_i2',frame_i2)
        frame_i1, frame_i2 = self.annos[frame_i1], self.annos[frame_i2]
        i_frame = frame_interpolation(frame_i1, frame_i2, frame_id)
        # self.update(frame_id,i_frame)
        return i_frame
    
    def make_box_obj(self,index,obj_info):
        boxes = []
        for obj in obj_info:
        
            x, y,z, yaw, obj_id, h, w, l, class_id = obj.tolist()
            
            if obj_id == -1.0:
                continue
            
            str_obj_id = str(int(obj_id))
            yaw = -yaw 

            if obj_id in [2.0,5.0,6.0]:
                # z+=0.06
                # x-=0.1

                yaw+=0.18
            
            if obj_id in [5.0,6.0]:
                 z-=0.4


            obj_center = np.array([x,y,z]) + SIZE_A
            obj_extent = EXP_RATE*  np.array([l,w,h]) 
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle([0, yaw, 0])
            quat = quaternion_from_matrix(rot)

            ply_path = self.lidar_path +f"/{str_obj_id}.ply"
            if not os.path.exists(ply_path):
                continue
           
            box = BoundingBox(obj_center,yaw = yaw, trackId=int(obj_id), size=obj_extent,
                        label=class_id, frame_id=index, frame=index, rot=rot, quat=quat)
            # box.transform(
            #     self.transform_matrix[:3, 3], self.transform_matrix[:3, :3])
            
            boxes.append(box)
            # use first box as meta
            if str_obj_id not in self.objects_meta:
                if self.lidar_path is not None:
                    pts = self.load_object_3D_points(str_obj_id)
                    if pts is None:
                        continue
                    self.seed_pts[str_obj_id] = pts
                    # CONSOLE.log(f"Loaded object_{str_obj_id}  {pts[0].shape} lidar points.")
                    self.objects_meta[str_obj_id] = box
                self.objects_frames[str_obj_id] = []
            self.objects_frames[str_obj_id].append(index)
        
        return boxes

                          
    def load_object_3D_points(self, obj_id: str):
        ply_path = self.lidar_path +f"/{obj_id}.ply"
        if not os.path.exists(ply_path):
            return None
        # assert ply_path.exists(), f"{ply_path} not exists"
        pcd = o3d.io.read_point_cloud(str(ply_path))
        # read points_xyz
        points3D = torch.from_numpy(np.array(pcd.points, dtype=np.float32))
        if points3D.shape[0] == 0:
        # if obj_id not in ['6','7','8','9']:
            return None
        # Load point6colours
        if pcd.has_colors():
            points3D_rgb = torch.from_numpy(np.array(pcd.colors, dtype=np.float32)).float() * 255.
        else:
            points3D_rgb = torch.rand(points3D.shape[0], 3, dtype=torch.float32) * 255.
            
        return (points3D, points3D_rgb)
