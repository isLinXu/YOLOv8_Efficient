# YOLOv8 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv8 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov8s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from ultralytics import YOLO


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source="0",  # ROOT / '0',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf=0.25,  # confidence threshold
        iou=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        # classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        # update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,  # High resolution masks
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)

    model = YOLO(weights)

    model.predict(source=source, iou=iou, conf=conf, save_txt=save_txt, save_conf=save_conf,
                  save_crop=save_crop, agnostic_nms=agnostic_nms, augment=augment,
                  visualize=visualize, project=project, name=name, exist_ok=exist_ok,
                  line_thickness=line_thickness, hide_labels=hide_labels, hide_conf=hide_conf, half=half, dnn=dnn,
                  vid_stride=vid_stride, retina_masks=retina_masks)

    # model.predict(source=source, view_img=True)  # Display preds. Accepts all yolo predict arguments
    # pt = model

    # Dataloader
    # bs = 1  # batch_size
    # if webcam:
    #     view_img = check_imshow(warn=True)
    #     dataset = LoadStreams(source, img_size=imgsz, auto=pt, vid_stride=vid_stride)
    #     bs = len(dataset)
    # elif screenshot:
    #     dataset = LoadScreenshots(source, img_size=imgsz, auto=pt)
    # else:
    #     dataset = LoadImages(source, img_size=imgsz, auto=pt, vid_stride=vid_stride)
    # vid_path, vid_writer = [None] * bs, [None] * bs


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov8n.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images/',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--view-img', default=True, action='store_true', help='View the prediction images')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--save-txt', default=False, action='store_true',
                        help='Save the results in a txt file')
    parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--save-conf', default=False, action='store_true', help='Save the condidence scores')
    parser.add_argument('--save-crop', default=False, action='store_true', help='Save the crop')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='Hide the labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='Hide the confidence scores')
    parser.add_argument('--vid-stride', default=False, action='store_true',
                        help='Input video frame-rate stride')
    parser.add_argument('--line-thickness', default=3, type=int, help='Bounding-box thickness (pixels)')
    parser.add_argument('--visualize', default=False, action='store_true', help='Visualize model features')
    parser.add_argument('--augment', default=False, action='store_true', help='Augmented inference')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--retina-masks', default=False, action='store_true', help='High resolution masks')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
