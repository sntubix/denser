# KITTI-MOT Dataset Organization

## Data Organization

The KITTI-MOT dataset should be organized as follows:

```
.(KITTI_MOT_ROOT)
└── training
    ├── calib
    │   └── sequence_id.txt
    ├── completion_02                # (Optional) depth completion
    │   └── sequence_id
    │       └── frame_id.png
    ├── completion_03                # (Optional) depth completion
    │   └── sequence_id
    │       └── frame_id.png
    ├── image_02
    │   └── sequence_id
    │       └── frame_id.png
    ├── image_03
    │   └── sequence_id
    │       └── frame_id.png
    ├── label_02
    │   └── sequence_id.txt
    ├── object_lidars
    │   └── object_id.txt
            ....
    └── oxts
        └── sequence_id.txt
```
