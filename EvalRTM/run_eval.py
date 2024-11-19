from eval_rtm import BinaryMetirc
from os import path as osp

from pathlib import Path
from glob import glob
from tqdm import tqdm

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

from prettytable import PrettyTable
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Full evaluation Metircs between binary masks')
    parser.add_argument('--pred_dir', help='pred mask dir')
    parser.add_argument('--gt_dir', help='save result in dir')
    parser.add_argument('--save_dir', help='save result in dir', default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    pred_dir = args.pred_dir

    if pred_dir[-1] == '/':
        pred_dir = pred_dir[:-1]

    mode = ['iou', 'f1']
    use_post = False

    gt_dir = args.gt_dir
    save_dir = args.save_dir


    method_name = osp.basename(pred_dir)
    method_name = method_name.split('_')[0]

    evaluation = BinaryMetirc(method_name, gt_dir, pred_dir, save_dir=save_dir, mode=mode)
    print("Evaluating [ {} ]".format(evaluation.method))


    evaluation.eval_full()
    evaluation.save_result()

    print("{} evaluation finished".format(evaluation.method))

    # evaluation.load_json(osp.join(evaluation.save_dir, 'result_{}.json'.format(method_name)))
    evaluation.show_result(mode=['iou', 'f1'])

    print("The result of [ {} ]".format(evaluation.method))

