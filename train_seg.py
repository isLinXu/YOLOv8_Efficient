import argparse


from ultralytics.yolo.utils.checks import print_args
from ultralytics.yolo.v8.detect import train
from ultralytics.yolo.engine.model import YOLO
from utils.callbacks import Callbacks

from utils.root_utils import PackageProjectUtil

ROOT = PackageProjectUtil.project_root_path()

def parse_opt():
    """
           CLI usage:
           python ultralytics/yolo/v8/detect/train.py model=yolov8n.yaml data=coco128 epochs=100 imgsz=640

           TODO:
           yolo task=detect mode=train model=yolov8n.yaml data=coco128.yaml epochs=100
       """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,default='seg', help='select train task, i.e.  detect or classify, seg')
    parser.add_argument('--mode', type=str,default='train' ,help='run mode')
    parser.add_argument('--model', type=str, default=ROOT + 'models/yolov8/seg/yolov8n-seg.yaml',help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT + 'data/coco128-seg.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')


    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt, callbacks=Callbacks()):
    trainer = YOLO(opt.model)
    trainer.train(task = opt.task, data=opt.data, epochs=opt.epochs,mode = opt.mode)


def run(**kwargs):
        # Usage: import train; train.run(data='coco128-seg.yaml', imgsz=320, weights='yolov8n-seg.pt')
        opt = parse_opt(True)
        for k, v in kwargs.items():
            setattr(opt, k, v)
        main(opt)
        return opt


class seg_cfg():
    def __init__(self, model, data, epochs):
        self.model = model
        self.data = data
        self.epochs = epochs

    def detect(self):
        return train(self)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)