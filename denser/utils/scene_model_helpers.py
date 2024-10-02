
import pywt

import torch
import numpy as np


def remove_prefix_from_keys(dictionary, prefix):
    new_dict = {}
    prefix_len = len(prefix)
    for key, value in dictionary.items():
        if key.startswith(prefix):
            new_key = key[prefix_len:]
        else:
            new_key = key
        new_dict[new_key] = value
    return new_dict

def get_bboxes(obj_info):
    bboxes = []
    for obj in obj_info[0]:
        if obj[4] >= 0:  # Valid object
            x, y, z, yaw, obj_id, l, w, h, class_id = obj
            bbox = {'x': x, 'y': y, 'z': z, 'yaw': yaw, 'l': l, 'h': h, 'w': w}
            bboxes.append(bbox)
    return bboxes

def is_within_bbox(point, bbox):
    # Transform point to bbox coordinate system
    cos_yaw = torch.cos(bbox['yaw'])
    sin_yaw = torch.sin(bbox['yaw'])
    dx = point[0] - bbox['x']
    dz = point[2] - bbox['z']
    x_local = cos_yaw * dx + sin_yaw * dz
    z_local = -sin_yaw * dx + cos_yaw * dz
    return (abs(x_local) <= bbox['l'] / 2) & (abs(point[1] - bbox['y']) <= bbox['h'] / 2) & (abs(z_local) <= bbox['w'] / 2)




def get_bboxes_pts(obj_info,tr_done=False):
    
    kitti2vkitti = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
    xs = []
    ys = []
    zs = []
    yaws = []

    for obj in obj_info[0]:
        if obj[4] >= 0:  # Valid object
            x, y, z, yaw, obj_id, l, w, h, class_id = obj
            
            if not tr_done:
                points3d = np.vstack((x, y, z)).T  # Transpose to shape (n, 3)
                # Convert points to homogeneous coordinates
                points3d_homogeneous = np.concatenate([points3d, np.ones((points3d.shape[0], 1))], axis=1)
                # points3d = (points3d_homogeneous @ opengl2kitti)#[...,:3].flatten()
                points3d = (points3d_homogeneous @ kitti2vkitti)[...,:3].flatten()
                x, y, z= points3d

            xs.append(x)
            ys.append(y)
            zs.append(z)
            yaws.append(yaw)
            
    xs= np.vstack(xs).flatten()
    ys =np.vstack(ys).flatten()
    zs =np.vstack(zs).flatten()
    yaws = np.vstack(zs).flatten()
    return xs,ys,zs,yaws



# Add Wavelet Transformation Functions
def SH2Wavelet(sh, wavelet='haar'):
    """
    Converts SH coefficients to wavelet coefficients.
    """
    coeffs = pywt.wavedec(sh.cpu().numpy(), wavelet, level=2)
    return [torch.tensor(coeff, device=sh.device, dtype=sh.dtype) for coeff in coeffs]

def Wavelet2SH(coeffs, wavelet='haar'):
    """
    Converts wavelet coefficients to SH coefficients.
    """
    sh = pywt.waverec([coeff.cpu().numpy() for coeff in coeffs], wavelet)
    return torch.tensor(sh, device=coeffs[0].device, dtype=coeffs[0].dtype)
