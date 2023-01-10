from pathlib import Path
import argparse, os, json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

parser = argparse.ArgumentParser(prog='get_coco_metric.py')
parser.add_argument("--annotation", type=str)
parser.add_argument("--prediction", type=str)
opt = parser.parse_args()

anno = opt.annotation
pred = opt.prediction

anno = COCO(anno)
pred = anno.loadRes(pred)
eval = COCOeval(anno, pred, 'bbox')
eval.evaluate()
eval.accumulate()
eval.summarize()
map, map50 = eval.stats[:2]