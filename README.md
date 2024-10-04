<div align="center"><h2>DENSER: 3D Gaussians Splatting for Scene Reconstruction of Dynamic Urban Environments</h2></div>

<div align="center"><h4>Scene Reconstruction Results</h4></div>

<div align="center"><h4>Ground Truth</h4></div>
<div align="center">
  <img alt="Ground Truth Output" src="./assets/videos/scene_0006_gt_output.gif" width="600px">
</div>

<div align="center"><h4>Reconstruction</h4></div>
<div align="center">
  <img alt="Reconstruction Output" src="./assets/videos/scene_0006_recoon_output.gif" width="600px">
</div>

<!-- <div align="center"><h4>Object Output</h4></div> -->
<div align="center">
  <img alt="Object Output" src="./assets/videos/scene_0006_obj_output.gif" width="600px">
</div>


## Installation

## Installing DENSER:

### Create environment
```bash
conda create --name denser -y python=3.8
conda activate denser
pip install --upgrade pip
```
### Clone the Repository
```bash
git clone https://github.com/mahmudi95/DENSER.git --recursive
```
### Install Dependencies
```bash
cd DENSER
bash installation.sh
```
### Install DENSER 
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


## Training,Rendering

```bash
ds-train denser --data /data/image_02/0006'

ds-render --load_config /path/to/your/config/0006/config.yml
```

## TODO

- write rendering script.
- Write evaluation script.
