# Yolov8_Efficient

![](./img/logo.png)

Simple and efficient use for yolov8


---

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/isLinXu/Yolov8_Efficient)  ![img](https://badgen.net/badge/icon/learning?icon=deepscan&label)
![](https://badgen.net/github/stars/isLinXu/Yolov8_Efficient)![](https://badgen.net/github/forks/isLinXu/Yolov8_Efficient)![](https://badgen.net/github/prs/isLinXu/Yolov8_Efficient)![](https://badgen.net/github/releases/isLinXu/Yolov8_Efficient)![](https://badgen.net/github/license/isLinXu/Yolov8_Efficient)![img](https://hits.dwyl.com/isLinXu/Yolov8_Efficient.svg)

## About

This is an unofficial repository maintained by independent developers for learning and communication based on the ultralytics v8 Weights and ultralytics Project.
If you have more questions and ideas, please feel free to discuss them together. In addition, if ultralytics releases the latest yolov8 warehouse, it is suggested to give priority to the official one.



## Performance

![](./img/demo.png)



## Quickstart

- **Documentation**

  [**Ultralytics YOLO Docs**](https://v8docs.ultralytics.com/sdk/)

- [ultralytics assets releases](https://github.com/ultralytics/assets/releases/)

### 1. CLI

To simply use the latest Ultralytics YOLO models

```bash
yolo task=detect    mode=train    model=yolov8n.yaml      args=...
          classify       predict        yolov8n-cls.yaml  args=...
          segment        val            yolov8n-seg.yaml  args=...
                         export         yolov8n.pt        format=onnx
```

### 2. Python SDK

To use pythonic interface of Ultralytics YOLO model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # create a new model from scratch
model = YOLO(
    "yolov8n.pt"
)  # load a pretrained model (recommended for best training results)
results = model.train(data="coco128.yaml", epochs=100, imgsz=640, ...)
results = model.val()
results = model.predict(source="bus.jpg")
success = model.export(format="onnx")
```

If you're looking to modify YOLO for R&D or to build on top of it, refer to [Using Trainer](<>) Guide on our docs.



### Pretrained Checkpoints

|                            Model                             | size (pixels) | mAPval 50-95 | mAPval 50 | Speed CPU b1 (ms) | Speed V100 b1 (ms) | Speed V100 b32 (ms) | params (M) | FLOPs @640 (B) |
| :----------------------------------------------------------: | :-----------: | :----------: | :-------: | :---------------: | :----------------: | :-----------------: | :--------: | :------------: |
| [yolov8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) |      640      |      -       |     -     |         -         |         -          |          -          |     -      |       -        |
| [yolov8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) |      640      |      -       |     -     |         -         |         -          |          -          |     -      |       -        |
| [yolov8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) |       -       |      -       |     -     |         -         |         -          |          -          |            |                |
| [yolov8-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) |       -       |      -       |     -     |         -         |         -          |          -          |     -      |       -        |
| [yolov8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) |       -       |      -       |     -     |         -         |         -          |          -          |     -      |       -        |
| [yolov8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) |       -       |      -       |     -     |         -         |         -          |          -          |     -      |       -        |
| [yolov8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) |       -       |      -       |     -     |         -         |         -          |          -          |     -      |       -        |
| [yolov8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) |       -       |      -       |     -     |         -         |         -          |          -          |     -      |       -        |
| [yolov8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) |       -       |      -       |     -     |         -         |         -          |          -          |     -      |       -        |
| [yolov8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) |       -       |      -       |     -     |         -         |         -          |          -          |     -      |       -        |

- TODO：Model testing and validation in progress



## Install

### pip install

```bash
pip install ultralytics
```

### Development

```shell
git clone git@github.com:isLinXu/YOLOv8_Efficient.git
cd YOLOv8_Efficient
cd ultralytics-master
pip install -e .
```

![](./img/install_img.png)

## Usage

### Train

- Single-GPU training:

```shell
python train.py --data coco128.yaml --weights weights/yolov8ns.pt --img 640  # from pretrained (recommended)
```

```python
python train.py --data coco128.yaml --weights '' --cfg yolov8ns.yaml --img 640  # from scratch
```

> Use IDE Pycharm
>
> ![](./img/pycharm_run_train.png)




  - Multi-GPU DDP training:
    
```shell
    python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov8ns.pt --img 640 --device 0,1,2,3
```

​    

### detect

```shell
python detect.py --weights yolov8s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

> Use IDE Pycharm
>
> ![](./img/pycharm_run_detect.png)



### val







## Acknowledgements

<details><summary> <b>Expand</b> </summary>
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
</details>
