import os
import argparse
import json
import pdb

import cv2
import numpy as np

import craft_utils
import imgproc
import file_utils

from craft import CRAFT
from refinenet import RefineNet

from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils import copyStateDict, str2bool


X_PADDING = 0.1
Y_PADDING = 0.005

# TEXT_REGION = [[30, 240], [600, 320]]    # ukraine
# TEXT_REGION = [[20, 470], [1220, 660]]   # russia
# TEXT_REGION = [[130, 530], [1140, 640]]  # obama
TEXT_REGION = [[75, 550], [1200, 700]]     # us
AREA_THRESH = 1000


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y, feature = net(x)

    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def load_refine_net(args):
    refine_net = RefineNet()
    if args.cuda:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
        refine_net = refine_net.cuda()
        refine_net = torch.nn.DataParallel(refine_net)
    else:
        refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

    refine_net.eval()

    return refine_net


def load_model(args):
    net = CRAFT()

    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    refine_net = None
    if args.refine:
        refine_net = load_refine_net(args)
        args.poly = True

    print("Loaded all models")
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
    # return True
    global TEXT_REGION, AREA_THRESH
    (x1, y1), (x2, y2) = box
    (xa, ya), (xb, yb) = TEXT_REGION
    iou = get_iou([x1, y1, x2, y2], [xa, ya, xb, yb])
    area = (x2 - x1) * (y2 - y1)
    return iou > 0 and area > AREA_THRESH


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


def main(image_list, net, refine_net, args, result_folder):
    print(f"Inferencing on {len(image_list)} images")
    for image_path in tqdm(image_list):

        image = imgproc.loadImage(image_path)
        cv2_img = cv2.imread(image_path)

        _, polys, score_text = test_net(net, image, args.text_threshold,
                                        args.link_threshold, args.low_text,
                                        args.cuda, args.poly, refine_net)

        filename = os.path.splitext(os.path.basename(image_path))[0]
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


if __name__ == '__main__':
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
    parser.add_argument('--refine', default=True, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')
    parser.add_argument('--test_folder', default='test_images/', type=str, help='folder path to input images')
    parser.add_argument('--result_folder', default='result/', type=str, help='folder for output images')

    args = parser.parse_args()
    net, refine_net = load_model(args)
    result_folder = args.result_folder
    image_list, _, _ = file_utils.get_files(args.test_folder)

    main(image_list, net, refine_net, args, result_folder)
    # USAGE:
    # python text_detection.py --test_folder "/data/chris/CRAFT-pytorch/us_air/us_frames_idx" --result_folder "/data/chris/CRAFT-pytorch/result/us_air"
    # python text_detection.py --test_folder "/data/chris/CRAFT-pytorch/frames/test/us_frames_idx" --result_folder "/data/chris/CRAFT-pytorch/result/test"
