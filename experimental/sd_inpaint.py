import os
import json

import torch
import cv2
import numpy as np
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline
from sklearn.cluster import KMeans


def combine_with_original(base_img, mask, result_img, output_path):
    mask = cv2.cvtColor(np.array(mask), cv2.COLOR_RGB2BGR)
    result_img = cv2.cvtColor(np.array(result_img), cv2.COLOR_RGB2BGR)
    final_img = base_img.copy()

    final_img[np.where(mask == 255)] = result_img[np.where(mask == 255)]
    cv2.imwrite(output_path, final_img)



def inpaint_with_model(result_folder, pipe, input_folder, params, max_bbox=False):
    # output_folder =
    parent_dir = os.path.dirname(result_folder)
    basename = os.path.basename(result_folder)

    image_folders_base = [f for f in os.listdir(result_folder) if os.path.isdir(os.path.join(result_folder, f))]
    image_folders = [os.path.join(result_folder, f) for f in image_folders_base]
    input_images = [os.path.join(input_folder, f"{f}.png") for f in image_folders_base]
    masks_folder = os.path.join(parent_dir, f"result_{basename}")
    os.makedirs(masks_folder, exist_ok=True)

    for i, folder in enumerate(image_folders):
        bbox_file = os.path.join(folder, "bboxes.json")
        base_img = cv2.imread(input_images[i])

        try:
            bboxes = json.load(open(bbox_file))["bbox"]
        except FileNotFoundError:
            print(f"No bounding boxes in {folder}")
            continue

        if len(bboxes) == 0:
            continue

        pil_base_img = Image.fromarray(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
        mask, masked_img = build_mask_from_bbox(bboxes, base_img)
        output_file = os.path.join(masks_folder, f"{image_folders_base[i]}.png")

        with torch.no_grad():
            result_image = pipe(image=masked_img, mask_image=mask, **params).images[0]
            result_image = result_image.resize(mask.size)
            combine_with_original(base_img, mask, result_image, output_file)


def build_mask_from_bbox(bboxes, base_img):
    height, width = base_img.shape[:2]
    masks = {}
    xmins = [bboxes[x][0][0] for x in bboxes]
    ymins = [bboxes[x][0][1] for x in bboxes]
    xmaxes = [bboxes[x][1][0] for x in bboxes]
    ymaxes = [bboxes[x][1][1] for x in bboxes]

    xmin_all, ymin_all = min(xmins), min(ymins)
    xmax_all, ymax_all = max(xmaxes), max(ymaxes)

    mask = np.zeros(base_img.shape, dtype=np.uint8)
    cv2.rectangle(mask, (xmin_all, ymin_all), (xmax_all, ymax_all), (255, 255, 255), -1)

    masked_img = mask.copy()
    masked_img[np.where(mask == 0)] = base_img[np.where(mask == 0)]

    masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)
    pil_mask = Image.fromarray(mask)
    pil_masked = Image.fromarray(masked_img)
    pil_mask.save("mask.png")
    pil_masked.save("masked.png")

    return mask, masked_img


def get_dominant_color_hist(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])

    dominant_hue_bin = np.argmax(hue_hist)

    dominant_color_bgr = np.array([[dominant_hue_bin * 2, 255, 255]], dtype=np.uint8)
    dominant_color_bgr = cv2.cvtColor(dominant_color_bgr, cv2.COLOR_HSV2BGR)

    return dominant_color_bgr


if __name__ == "__main__":
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
