import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from tqdm import tqdm
import torch
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import cv2
import json
import pdb
from sklearn.cluster import KMeans


from PIL import Image

torch.manual_seed(0)

X_PADDING = 10
Y_PADDING = 10


def get_dominant_color_kmeans(image):
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pixels)

    labels, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    dominant_label = labels[np.argmax(cluster_sizes)]

    dominant_color = kmeans.cluster_centers_[dominant_label]
    dominant_color = dominant_color.astype(int)

    # dominant_color_bgr = dominant_color[::-1]
    return dominant_color


def get_max_bbox(bboxes, height, width):
    xmins = [bboxes[x][0][0] for x in bboxes]
    ymins = [bboxes[x][0][1] for x in bboxes]
    xmaxes = [bboxes[x][1][0] for x in bboxes]
    ymaxes = [bboxes[x][1][1] for x in bboxes]

    xmin_all, ymin_all = max(min(xmins) - X_PADDING, 0), max(min(ymins) - Y_PADDING, 0)
    xmax_all, ymax_all = min(max(xmaxes) + X_PADDING, width), min(max(ymaxes) + Y_PADDING, height)

    return xmin_all, ymin_all, xmax_all, ymax_all


def blur_textarea(bboxes, base_img, blur_type="color_based_blur", max_bbox=False):
    height, width = base_img.shape[:2]
    masks = {}

    if max_bbox:
        xmin_all, ymin_all, xmax_all, ymax_all = bboxes
    else:
        xmin_all, ymin_all, xmax_all, ymax_all = get_max_bbox(bboxes, height, width)

    if blur_type == "color_based_blur":
        bbox_area = base_img[ymin_all:ymax_all, xmin_all:xmax_all]

        dom_kmeans = get_dominant_color_kmeans(bbox_area)
        # dom_hist = get_dominant_color_hist(bbox_area)

        b, g, r = dom_kmeans
        mask = base_img.copy()
        cv2.rectangle(mask, (xmin_all, ymin_all), (xmax_all, ymax_all), (int(b), int(g), int(r)), cv2.FILLED)

        alpha = 200
        alpha_percentage = alpha / 255.0
        blurred_img = cv2.addWeighted(base_img, 1 - alpha_percentage, mask, alpha_percentage, 0)

        for i in range(5):
            blurred_img = cv2.GaussianBlur(blurred_img, (15, 15), 0)

        result = base_img.copy()
        result[ymin_all:ymax_all, xmin_all:xmax_all] = blurred_img[ymin_all:ymax_all, xmin_all:xmax_all]

        # cv2.imwrite("mask.png", mask)
        # cv2.imwrite("masked.png", result)

    return mask, result


def inpaint_with_opencv(result_folder, input_folder, max_bbox=False):
    # output_folder =
    parent_dir = os.path.dirname(result_folder)
    basename = os.path.basename(result_folder)

    image_folders_base = [f for f in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, f))]
    image_folders = [os.path.join(result_folder, f) for f in image_folders_base]
    input_images = [os.path.join(input_folder, f"{f}.png") for f in image_folders_base]
    # masks_folder = os.path.join(parent_dir, f"result_{basename}")
    # os.makedirs(masks_folder, exist_ok=True)

    ocr_transcript = os.path.join(result_folder, "corrected.json")

    if max_bbox:
        max_area = 0
        max_box = [0, 530, 1280, 720]

        # for i, folder in enumerate(image_folders):
        #     bbox_file = os.path.join(folder, "bboxes.json")
        #     base_img = cv2.imread(input_images[i])
        #     height, width = base_img.shape[:2]
        #     try:
        #         bboxes = json.load(open(bbox_file))["bbox"]
        #     except FileNotFoundError:
        #         print(f"No bounding boxes in {folder}")
        #         continue

        #     xmin_all, ymin_all, xmax_all, ymax_all = get_max_bbox(bboxes, height, width)
        #     area = (xmax_all - xmin_all) * (ymax_all - ymin_all)
        #     if area > max_area:
        #         max_area = area
        #         max_box = (xmin_all, ymin_all, xmax_all, ymax_all)

        for i, folder in enumerate(tqdm(image_folders)):
            base_img = cv2.imread(input_images[i])
            mask, blurred = blur_textarea(max_box, base_img, max_bbox=max_bbox)
            output_file = os.path.join(folder, "masked.png")
            cv2.imwrite(output_file, blurred)

    else:
        for i, folder in enumerate(tqdm(image_folders)):
            bbox_file = os.path.join(folder, "bboxes.json")
            base_img = cv2.imread(input_images[i])

            try:
                bboxes = json.load(open(bbox_file))["bbox"]
            except FileNotFoundError:
                print(f"No bounding boxes in {folder}")
                continue

            if len(bboxes) == 0:
                continue

            mask, blurred = blur_textarea(bboxes, base_img, max_bbox)
            output_file = os.path.join(folder, "masked.png")
            cv2.imwrite(output_file, blurred)


if __name__ == "__main__":
    max_bbox = True

    input_folder = "/data/chris/CRAFT-pytorch/us_air/us_frames_idx"
    result_folder = "/data/chris/CRAFT-pytorch/result/us_frames_idx"

    # input_folder = "/data/chris/CRAFT-pytorch/obama_india/obama_frames_idx"
    # result_folder = "/data/chris/CRAFT-pytorch/result/obama_frames_idx"

    # input_folder = "/data/chris/CRAFT-pytorch/russia_police/russia_frames_idx"
    # result_folder = "/data/chris/CRAFT-pytorch/result/russia_frames_idx"

    # input_folder = "/data/chris/CRAFT-pytorch/ukraine_losses/raw_frames_idx"
    # result_folder = "/data/chris/CRAFT-pytorch/result/raw_frames_idx"

    inpaint_with_opencv(result_folder, input_folder, True)

    pdb.set_trace()

# transformers==4.18.0
