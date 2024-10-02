# DENSER

## Installation

## Installing DENSER:

### Step 1: Clone the Repository
```bash
git clone https://github.com/mahmudi95/DENSER.git --recursive
```
### Step 2: Install Dependencies
```bash
cd DENSER
bash installation.sh
```
### Step 3: Install DENSER 
```bash
pip install -e .
```
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
