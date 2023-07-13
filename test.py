"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import json
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

#
import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='test_images/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

X_PADDING = 0.1
Y_PADDING = 0.005

# text_region = [[30, 240], [600, 320]]
# text_region = [[20, 470], [1220, 660]]   #
# text_region = [[130, 530], [1140, 640]]   # obama
text_region = [[75, 550], [1200, 700]]   # us
area_threshold = 1000


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


def split_image_polygons(img, polys, output_folder):
    for idx, polygon in enumerate(polys):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [polygon], (255, 255, 255))  # Fill the polygon with white
        cropped_image = cv2.bitwise_and(img, mask)
        x, y, w, h = cv2.boundingRect(polygon)
        cropped_image = cropped_image[y:y+h, x:x+w]
        cv2.imwrite(f'cropped_{idx}.jpg', cropped_image)


def load_model(args):
    net = CRAFT()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    print("Loaded all weights")

    return net, refine_net


def get_bboxes(polys, height, width):
    bboxes = []
    for poly in polys:
        xs = poly[:, 0]
        ys = poly[:, 1]

        xmin, xmax = max(0, xs.min()), min(width, xs.max())
        ymin, ymax = max(0, ys.min()), min(height, ys.max())

        bbox = [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]
        bboxes.append(bbox)

    return bboxes


def get_iou(box1, box2):
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    x_intersection = max(box1[0], box2[0])
    y_intersection = max(box1[1], box2[1])
    w_intersection = min(box1[2], box2[2]) - x_intersection + 1
    h_intersection = min(box1[3], box2[3]) - y_intersection + 1
    area_intersection = max(0, w_intersection) * max(0, h_intersection)
    iou = area_intersection / (area_box1 + area_box2 - area_intersection)
    return iou


def check_valid_bbox(box):
    global text_region, area_threshold
    (x1, y1), (x2, y2) = box
    (xa, ya), (xb, yb) = text_region
    iou = get_iou([x1, y1, x2, y2], [xa, ya, xb, yb])
    area = (x2 - x1) * (y2 - y1)
    return iou > 0 and area > area_threshold


def write_polygons(polys, cv2_img, output_folder):
    global X_PADDING, Y_PADDING
    height, width, _ = cv2_img.shape

    bboxes = get_bboxes(polys, height, width)
    result = {}
    padded = {}
    for i, box in enumerate(bboxes):
        fname = f"cropped_{i}.png"
        if check_valid_bbox(box):
            result[fname] = box
            (xmin, ymin), (xmax, ymax) = box
            if X_PADDING > 0:
                xmin, xmax = max(0, xmin - int(width*X_PADDING)), min(width, xmax + int(width*X_PADDING))
            if Y_PADDING > 0:
                ymin, ymax = max(0, ymin - int(height*Y_PADDING)), min(height, ymax + int(height*Y_PADDING))

            padded[fname] = [[xmin, ymin], [xmax, ymax]]
            cropped_img = cv2_img[ymin:ymax, xmin:xmax]
            # pdb.set_trace()
            cv2.imwrite(os.path.join(output_folder, fname), cropped_img)

    return result, padded


if __name__ == '__main__':
    net, refine_net = load_model(args)
    # """ For test images in a folder """
    result_folder = 'result/'

    image_list, _, _ = file_utils.get_files("/data/chris/CRAFT-pytorch/us_air/us_frames_idx")
    # image_list, _, _ = file_utils.get_files("/data/chris/CRAFT-pytorch/russia_police/russia_frames_idx")
    # image_list, _, _ = file_utils.get_files(args.test_folder)
    # pdb.set_trace()
    # load data
    print("Inferencing on images")
    for k, image_path in enumerate(tqdm(image_list)):
        # print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        # pdb.set_trace()
        image = imgproc.loadImage(image_path)
        cv2_img = cv2.imread(image_path)

        bboxes, polys, score_text = test_net(net, image, args.text_threshold,
                                             args.link_threshold, args.low_text,
                                             args.cuda, args.poly, refine_net)

        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        parent_folder = os.path.basename(os.path.dirname(image_path))
        output_folder = os.path.join(result_folder, parent_folder, filename)
        os.makedirs(output_folder, exist_ok=True)

        heatmap_file = os.path.join(output_folder, f"res_{filename}_heat.jpg")
        cv2.imwrite(heatmap_file, score_text)

        # if k==2:
        result, padded = write_polygons(polys, cv2_img, output_folder)
        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=output_folder)

        json.dump({"bbox": result, "padded": padded},
                  open(os.path.join(output_folder, "bboxes.json"), "w"), indent=4)
