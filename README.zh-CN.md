# Yolov8_Efficient

![](./img/logo.png)

简单高效的使用YOLOv8

[English](README.md) | [简体中文](README.zh-CN.md)

---

[![Build Status](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Factions-badge.atrox.dev%2Fatrox%2Fsync-dotenv%2Fbadge&style=flat)](https://github.com/isLinXu/Yolov8_Efficient)  ![img](https://badgen.net/badge/icon/learning?icon=deepscan&label)
![](https://badgen.net/github/stars/isLinXu/Yolov8_Efficient)![](https://badgen.net/github/forks/isLinXu/Yolov8_Efficient)![](https://badgen.net/github/prs/isLinXu/Yolov8_Efficient)![](https://badgen.net/github/releases/isLinXu/Yolov8_Efficient)![](https://badgen.net/github/license/isLinXu/Yolov8_Efficient)![img](https://hits.dwyl.com/isLinXu/Yolov8_Efficient.svg)

## 😎 介绍

这是一个由独立开发人员维护的非官方存储库，用于基于 ultralytics v8 Weights 和 ultralytics Project 的学习和交流。如果大家有更多的问题和想法，欢迎大家一起讨论。
另外，ultralytics已发布了最新的[ultralytics](https://github.com/ultralytics/ultralytics)仓库，建议优先使用官方的。



## 🥰展示

### 指标

![](./img/plot_metrics.jpg)

### 网络结构图 

![](./img/v8_structure.jpg)

> 这里感谢[集智书童](https://github.com/jizhishutong)为本项目提供的网络结构图

![](./img/demo.png)

- wandb训练日志:  [log](https://wandb.ai/glenn-jocher/YOLOv8)
- 实验日志: [log](https://github.com/isLinXu/YOLOv8_Efficient/tree/main/log)



## 🆕新闻!

---

- ... ...
- 2023/01/11 - add metrics plot and model structure
- 2023/01/10 - add yolov8 metrics and logs
- 2023/01/09 - add val.py and fix some error
- 2023/01/07 - fix some error and warning 
- 2023/01/06 - add train.py, detect.py and README.md
- 2023/01/06 - Create and Init a new repository



## 🤔 任务清单：

- [ ] 模型测试和验证中
- [ ] 



## 🧙‍快速开始

- **文档**

  [**Ultralytics YOLO Docs**](https://v8docs.ultralytics.com/sdk/)

- [ultralytics assets releases](https://github.com/ultralytics/assets/releases/)


### 1.命令行执行

简单地使用最新的 Ultralytics YOLO 模型

```bash
yolo task=detect    mode=train    model=yolov8n.yaml      args=...
          classify       predict        yolov8n-cls.yaml  args=...
          segment        val            yolov8n-seg.yaml  args=...
                         export         yolov8n.pt        format=onnx
```

### 2. Python SDK

使用 Ultralytics YOLO 模型的 pythonic 接口

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

如果您希望为研发修改 YOLO 或在其之上构建，请参阅我们文档中的[Using Trainer](https://docs.ultralytics.com/)指南。。



## 🧙‍预训练检查点

|                            Model                             | size (pixels) | mAPval 50-95 | mAPval 50 | Speed CPU b1 (ms) | Speed RTX 3080 b1(ms) | layers | params (M) | FLOPs @640 (B) |
| :----------------------------------------------------------: | :-----------: | :----------: | :-------: | :---------------: | :-------------------: | :----: | :--------: | :------------: |
| [yolov8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) |      640      |     37.2     |   53.2    |       47.2        |          5.6          |  168   |    3.15    |      8.7       |
| [yolov8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) |      640      |     30.7     |   50.0    |       59.3        |         11.3          |  195   |    3.40    |      12.6      |
| [yolov8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) |      640      |     44.7     |   62.2    |       87.9        |          5.7          |  168   |   11.15    |      28.6      |
| [yolov8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) |      640      |     37.0     |   58.8    |       107.6       |         11.4          |  195   |   11.81    |      42.6      |
| [yolov8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) |      640      |     49.9     |   67.4    |       185.6       |          8.3          |  218   |   25.89    |      78.9      |
| [yolov8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) |      640      |     40.6     |   63.5    |       207.7       |         15.3          |  245   |   27.27    |     110.2      |
| [yolov8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) |      640      |     52.4     |   69.9    |       319.6       |         13.1          |  268   |   43.67    |     165.2      |
| [yolov8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) |      640      |     42.5     |   66.1    |       296.9       |         16.8          |  295   |   45.97    |     220.5      |
| [yolov8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) |      640      |     53.5     |   70.9    |       334.6       |         20.4          |  268   |   68.20    |     257.8      |
| [yolov8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) |      640      |     43.2     |   67.1    |       418.8       |         23.8          |  295   |   71.80    |     344.1      |

> - **表格注释** 以上数据是在以下配置的环境中运行测试生成的。详情见下文。
> - 显卡：NVIDIA GeForce RTX 3080/PCIe/SSE2
> - CPU：Intel® Core™ i9-10900K CPU @ 3.70GHz × 20
> - 内存：31.3 GiB
> - 系统：Ubuntu 18.04 LTS
> - (ms): 这里的统计速度是推理速度



## 安装

### 直接安装

```bash
pip install ultralytics
```

### 配置安装

```shell
git clone git@github.com:isLinXu/YOLOv8_Efficient.git
cd YOLOv8_Efficient
cd ultralytics-master
pip install -e .
```

![](./img/install_img.png)

## 🔨用法

### 训练

- 单 GPU 训练:

```shell
python train.py --data coco128.yaml --weights weights/yolov8ns.pt --img 640  # from pretrained (recommended)
```

```python
python train.py --data coco128.yaml --weights '' --cfg yolov8ns.yaml --img 640  # from scratch
```

> 使用 IDE Pycharm
>
> ![](./img/pycharm_run_train.png)




  - 多 GPU DDP 训练：
```shell
    python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov8ns.pt --img 640 --device 0,1,2,3
```

​    

### 推理检测

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

> 使用 IDE Pycharm
>
> ![](./img/pycharm_run_detect.png)



### 验证

- 以coco128为例:

>| ![](./img/val_batch1_pred.jpg) | ![](./img/F1_curve.png) | ![](./img/P_curve.png)          |
> | ------------------------------ | ----------------------- | ------------------------------- |
>| ![](./img/PR_curve.png)        | ![](./img/R_curve.png)  | ![](./img/confusion_matrix.png) |

#### 用法:

```shell
python val.py --weights yolov8n.pt --data coco128.yaml --img 640
```

#### 用法 - 格式:


```shell
python val.py --weights yolov8s.pt                 # PyTorch
                              yolov8s.torchscript        # TorchScript
                              yolov8s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov8s_openvino_model     # OpenVINO
                              yolov8s.engine             # TensorRT
                              yolov8s.mlmodel            # CoreML (macOS-only)
                              yolov8s_saved_model        # TensorFlow SavedModel
                              yolov8s.pb                 # TensorFlow GraphDef
                              yolov8s.tflite             # TensorFlow Lite
                              yolov8s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8s_paddle_model       # PaddlePaddle
```

> 使用 IDE Pycharm
> ![](./img/pycharm_run_val.png)





## 🌹致谢
- [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
- [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- https://github.com/meituan/YOLOv6
- https://github.com/WongKinYiu/yolov7

