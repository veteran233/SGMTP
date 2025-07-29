<center><p style="font-size:x-large;font-weight:bold">SGMTP : A new tools for test prioritization of LiDAR-based autonomous vehicle tracking</p></center>

---

This is the official implementation of paper "**Test Prioritization for LiDAR-based Object Tracking Systems via Scene Graph**".

## Environment

#### Our working environment

- Ubuntu 20.04
- Python 3.7
- PyTorch 1.13.1
- CUDA 11.7

#### Dependency

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)

OpenPCDet is a clear, simple, self-contained open source project for LiDAR-based 3D object detection. Please [click here](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) for the installation of OpenPCDet.

- [TrackEval](https://github.com/JonathonLuiten/TrackEval)

HOTA (and other) evaluation metrics for Multi-Object Tracking (MOT). Installation of TrackEval :
```
cd path/to/TrackEval
python setup.py develop
```

- Other python package :
```
pip install -r requirements.txt
```

## Usage

#### KITTI Dataset

Download KITTI Object Tracking Dataset : [click here](https://www.cvlibs.net/datasets/kitti/eval_tracking.php).

This repository require :
1. velodyne point clouds (35 GB)
2. camera calibration matrices of tracking data set (1 MB)
3. training labels (9 MB)

#### Pose

Download pose (from [3D-Multi-Object-Tracker](https://github.com/hailanyi/3D-Multi-Object-Tracker)) : [click here](https://drive.google.com/drive/folders/1Vw_Mlfy_fJY6u0JiCD-RMb6_m37QAXPQ?usp=sharing).

#### Scene Graph

Download scene graph : 

After completing the download, please organize the files according to the following file tree structure and place them in any location that is convenient:

`kitti_tracking_dataset` folder tree
```
kitti_tracking_dataset
├── testing
│   ├── calib
│   ├── pose
│   ├── sg
│   └── velodyne
└── training
    ├── calib
    ├── label_02
    ├── pose
    ├── sg
    └── velodyne
```

#### Start

0. Ensure that you are in the root directory of the repository : ```cd /path/to/SGMTP/```

1. Please type the command : ```python tools/generate_dataset.py -odp /path/to/kitti_tracking_dataset```

The parameters of ```generate_dataset.py```
```
usage: generate_dataset.py [-h] -odp ORIGIN_DATASET_PATH [-s {val,test}]
                           [-o OUTPUT_PATH]
                           [--second-per-data SECOND_PER_DATA]
                           [--stride STRIDE]
                           [--scene-exclude SCENE_EXCLUDE [SCENE_EXCLUDE ...]]

optional arguments:
  -h, --help            show this help message and exit
  -odp ORIGIN_DATASET_PATH, --origin-dataset-path ORIGIN_DATASET_PATH
                        Path to the origin dataset
  -s {val,test}, --split {val,test}
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
  --second-per-data SECOND_PER_DATA
                        Time duration (in seconds)
  --stride STRIDE       Step size/window shift (in seconds)
  --scene-exclude SCENE_EXCLUDE [SCENE_EXCLUDE ...]
                        List of scenes to exclude
```

2. Please type the command : ```python step1.py```

The parameters of ```step1.py```
```
usage: step1.py [-h] [-dp DATASET_PATH] [-s {val,test}]
                [-m MODELS_NAME [MODELS_NAME ...]] [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -dp DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset
  -s {val,test}, --split {val,test}
  -m MODELS_NAME [MODELS_NAME ...], --models-name MODELS_NAME [MODELS_NAME ...]
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
```

3. Please type the command : ```python step2.py```

The parameters of ```step2.py```
```
usage: step2.py [-h] [-fw FRAMEWORKS [FRAMEWORKS ...]]
                [-det DETECTORS [DETECTORS ...]] [-dp DATASET_PATH]
                [-detp DETECTIONS_PATH] [-s {val,test}] [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -fw FRAMEWORKS [FRAMEWORKS ...], --frameworks FRAMEWORKS [FRAMEWORKS ...]
  -det DETECTORS [DETECTORS ...], --detectors DETECTORS [DETECTORS ...]
  -dp DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset
  -detp DETECTIONS_PATH, --detections-path DETECTIONS_PATH
                        Path to the detections result
  -s {val,test}, --split {val,test}
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
```

4. Please type the command : ```python step3.py```

The parameters of ```step3.py```
```
usage: step3.py [-h] [-fw FRAMEWORKS [FRAMEWORKS ...]]
                [-det DETECTORS [DETECTORS ...]] [-dp DATASET_PATH]
                [-tp TRACKING_PATH] [-s {val,test}] [-o OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -fw FRAMEWORKS [FRAMEWORKS ...], --frameworks FRAMEWORKS [FRAMEWORKS ...]
  -det DETECTORS [DETECTORS ...], --detectors DETECTORS [DETECTORS ...]
  -dp DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset
  -tp TRACKING_PATH, --tracking-path TRACKING_PATH
                        Path to the trackers result
  -s {val,test}, --split {val,test}
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
```

#### For RQ1

RQ1 seeks to the effectiveness that SGMTP perform on test prioritization tasks.

Please type the command : ```python RQ1.py```

The parameters of ```RQ1.py```
```
usage: RQ1.py [-h] [-dp DATASET_PATH] [-fw FRAMEWORKS [FRAMEWORKS ...]]
              [-det DETECTORS [DETECTORS ...]] [-tp TEST_PATH] [-s {val,test}]
              [-c CRITERIA [CRITERIA ...]]

optional arguments:
  -h, --help            show this help message and exit
  -dp DATASET_PATH, --dataset-path DATASET_PATH
                        Path to the dataset
  -fw FRAMEWORKS [FRAMEWORKS ...], --frameworks FRAMEWORKS [FRAMEWORKS ...]
  -det DETECTORS [DETECTORS ...], --detectors DETECTORS [DETECTORS ...]
  -tp TEST_PATH, --test-path TEST_PATH
                        Path to the test prioritization result
  -s {val,test}, --split {val,test}
  -c CRITERIA [CRITERIA ...], --criteria CRITERIA [CRITERIA ...]
```

#### For RQ2

RQ2 seeks to the effectiveness of SGMTP in error detection across different types of error scene graph.

Please type the command : ```python RQ2.py```