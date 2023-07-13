import os

import datetime
from tqdm import tqdm
import numpy as np
import cv2
import json
import pdb
from PIL import Image, ImageDraw, ImageFont

from collections import defaultdict


X_PADDING = 10
Y_PADDING = 10
MIN_SENTENCE_LENGTH = 8

fonts = {
    "hi-IN": "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
    "bn-IN": "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
    "gu-IN": "/usr/share/fonts/truetype/lohit-gujarati/Lohit-Gujarati.ttf",
    "mr-IN": "/usr/share/fonts/truetype/lohit-marathi/Lohit-Marathi.ttf",
    "te-IN": "/usr/share/fonts/truetype/lohit-telugu/Lohit-Telugu.ttf",
    "ta-IN": "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Tamil.ttf",
    "kn-IN": "/usr/share/fonts/truetype/lohit-kannada/Lohit-Kannada.ttf",
    "ml-IN": "/usr/share/fonts/truetype/lohit-malayalam/Lohit-Malayalam.ttf"
}

langs = {
        "hi-IN": "hindi",
        "bn-IN": "bengali",
        "gu-IN": "gujarati",
        "mr-IN": "marathi",
        "ta-IN": "tamil",
        "te-IN": "telugu",
        "kn-IN": "kannada",
        "ml-IN": "malayalam"
}


def get_total_bbox(bboxes, width=None, height=None, pad=True):
    xmins = [bboxes[x][0][0] for x in bboxes]
    ymins = [bboxes[x][0][1] for x in bboxes]
    xmaxes = [bboxes[x][1][0] for x in bboxes]
    ymaxes = [bboxes[x][1][1] for x in bboxes]

    if pad:
        xmin_all, ymin_all = max(min(xmins) - X_PADDING, 0), max(min(ymins) - Y_PADDING, 0)
        xmax_all, ymax_all = min(max(xmaxes) + X_PADDING, width), min(max(ymaxes) + Y_PADDING, height)
    else:
        xmin_all, ymin_all = min(xmins), min(ymins)
        xmax_all, ymax_all = max(xmaxes), max(ymaxes)

    return xmin_all, ymin_all, xmax_all, ymax_all


def get_time_in_seconds(timestamp):
    time_format = '%H:%M:%S.%f'
    time_object = datetime.datetime.strptime(timestamp, time_format)
    total_seconds = time_object.hour * 3600 + time_object.minute * 60 + time_object.second + time_object.microsecond / 1e6
    return total_seconds


def get_region_with_text(timestamps, transcripts, image_ids, fps, image_folders):
    selected_ids = []
    for (st, et), text in zip(timestamps, transcripts):
        st, et = get_time_in_seconds(st), get_time_in_seconds(et)
        start_id, end_id = int((st + 1) * fps), int((et + 1) * fps)

        subset_image_ids = [(i, x, y) for i, (x, y) in enumerate(zip(image_ids, image_folders))
                            if x >= start_id and x < end_id]
        valid_image_ids = [(i, x) for i, x, y in subset_image_ids if os.path.exists(os.path.join(y, "bboxes.json"))]

        if len(valid_image_ids) == 0:
            selected_id = [(None, (start_id + end_id) // 2, text)]
        else:
            selected_id = [x + (text, ) for x in valid_image_ids]
            # selected_id = valid_image_ids[len(valid_image_ids) // 2]
        selected_ids.extend(selected_id)
    # pdb.set_trace()
    return selected_ids


def get_region_per_frame(regions, total_frames, include_blanks=False):
    valid_regions = [x for x in regions if len(x[2].strip()) > 0]
    frame_ids = list(range(total_frames))

    if include_blanks:
        nearest_regions = [min(regions, key=lambda x: abs(x[1]-frame_id))
                           for frame_id in frame_ids]
        return regions, nearest_regions
    else:
        nearest_regions = [min(valid_regions, key=lambda x: abs(x[1]-frame_id))
                           for frame_id in frame_ids]
        return valid_regions, nearest_regions


def divide_list(text_list, n):
    quotient, remainder = divmod(len(text_list), n)
    sublists = []
    start = 0

    for i in range(n):
        sublist_size = quotient + 1 if i < remainder else quotient
        sublists.append(text_list[start:start+sublist_size])
        start += sublist_size

    return sublists


def add_transcripts_to_masked(regions, image_folders, lang, max_box):
    inpainted_images = []
    for (img_id, frame_id, text) in tqdm(regions):
        if len(text.strip()) == 0:
            continue
        bboxes = json.load(open(os.path.join(image_folders[img_id], "bboxes.json")))["bbox"]
        text_list = text.split()
        divided_text = divide_list(text_list, len(bboxes))
        boxes = sorted(list(bboxes.values()), key=lambda x:x[0][1])

        max_box = get_total_bbox(bboxes, pad=False)
        max_box_x = max_box[0]
        max_box_width = max_box[2] - max_box[0]

        img_pil = Image.open(os.path.join(image_folders[img_id], "masked.png"))
        draw = ImageDraw.Draw(img_pil)

        font_size = 24
        font_color = (255, 255, 255)
        fontpath = fonts[lang]
        font = ImageFont.truetype(fontpath, 32)

        scales = []
        for box, text_list in zip(boxes, divided_text):
            text_str = " ".join(text_list)

            (bbox_x, bbox_y), (xm, ym) = box
            bbox_w = xm - bbox_x
            bbox_h = ym - bbox_y

            text_width, text_height = draw.textsize(text_str, font=font)
            # scale = min(bbox_w / text_width, bbox_h / text_height)
            scale = max_box_width / text_width

            scales.append(scale)

        adjusted_font_size = int(font_size * min(scales))
        # print(adjusted_font_size)
        # if frame_id == 2100:
        #     adjusted_font_size *= 2

        for text_list, box in zip(divided_text, boxes):
            text_str = " ".join(text_list)
            adjusted_font = ImageFont.truetype(fontpath, adjusted_font_size)

            (bbox_x, bbox_y), (xm, ym) = box
            bbox_w = xm - bbox_x
            bbox_h = ym - bbox_y

            new_text_width, new_text_height = draw.textsize(text_str, font=adjusted_font)


            # x_pos = bbox_x + (bbox_w - new_text_width) / 2
            x_pos = max_box_x + (max_box_width - new_text_width) / 2
            y_pos = bbox_y + (bbox_h - new_text_height) / 2
            draw.text((x_pos, y_pos), text_str, font=adjusted_font, fill=font_color)

        output_path = os.path.join(image_folders[img_id], f"inpainted-{lang}.png")
        img_pil.save(output_path)
        inpainted_images.append(output_path)

    return inpainted_images



def divide_box(box, num_boxes):
    xmin, ymin, xmax, ymax = box
    start = ymin
    end = ymax
    numbers = np.linspace(start, end, num_boxes+1).astype(int).tolist()
    boxes = [(xmin, numbers[i], xmax, numbers[i+1]) for i in range(len(numbers)-1)]
    return boxes


def add_transcripts_to_masked_max_bbox(regions, image_folders, lang, max_box=[0, 530, 1280, 720], num_lines=2):
    inpainted_images = []
    font_sizes = []
    for (img_id, frame_id, text) in tqdm(regions):
        if len(text.strip()) == 0:
            continue

        text_list = text.split()

        boxes = divide_box(max_box, num_lines)
        divided_text = divide_list(text_list, len(boxes))

        img_pil = Image.open(os.path.join(image_folders[img_id], "masked.png"))
        draw = ImageDraw.Draw(img_pil)

        font_size = 24
        font_color = (255, 255, 255)
        fontpath = fonts[lang]
        font = ImageFont.truetype(fontpath, 32)

        scales = []
        for box, text_list in zip(boxes, divided_text):
            text_str = " ".join(text_list)
            bbox_x, bbox_y, xm, ym = box
            bbox_w = xm - bbox_x
            bbox_h = ym - bbox_y

            text_width, text_height = draw.textsize(text_str, font=font)
            scale = min(bbox_w / text_width, bbox_h / text_height)
            # scale = max_box_width / text_width
            scales.append(scale)

        adjusted_font_size = int(font_size * min(scales))
        font_sizes.append(adjusted_font_size)

    adjusted_font_size = min(font_sizes)

    for (img_id, frame_id, text) in tqdm(regions):
        if len(text.strip()) == 0:
            continue

        text_list = text.split()

        boxes = divide_box(max_box, num_lines)
        divided_text = divide_list(text_list, len(boxes))

        img_pil = Image.open(os.path.join(image_folders[img_id], "masked.png"))
        draw = ImageDraw.Draw(img_pil)

        # font_size = 24
        font_color = (255, 255, 255)
        fontpath = fonts[lang]
        # font = ImageFont.truetype(fontpath, 32)

        # scales = []
        # for box, text_list in zip(boxes, divided_text):
        #     text_str = " ".join(text_list)
        #     bbox_x, bbox_y, xm, ym = box
        #     bbox_w = xm - bbox_x
        #     bbox_h = ym - bbox_y

        #     text_width, text_height = draw.textsize(text_str, font=font)
        #     scale = min(bbox_w / text_width, bbox_h / text_height)
        #     # scale = max_box_width / text_width
        #     scales.append(scale)

        # adjusted_font_size = int(font_size * min(scales))

        for text_list, box in zip(divided_text, boxes):
            text_str = " ".join(text_list)
            adjusted_font = ImageFont.truetype(fontpath, adjusted_font_size)

            bbox_x, bbox_y, xm, ym = box
            bbox_w = xm - bbox_x
            bbox_h = ym - bbox_y

            new_text_width, new_text_height = draw.textsize(text_str, font=adjusted_font)

            x_pos = bbox_x + (bbox_w - new_text_width) / 2
            # x_pos = max_box_x + (max_box_width - new_text_width) / 2
            y_pos = bbox_y + (bbox_h - new_text_height) / 2
            draw.text((x_pos, y_pos), text_str, font=adjusted_font, fill=font_color)

        output_path = os.path.join(image_folders[img_id], f"inpainted-{lang}.png")
        img_pil.save(output_path)
        inpainted_images.append(output_path)

    return inpainted_images


def build_video_max_bbox(video, valid_regions, regions_per_frame, image_folders, fps,
                         width, height, total_frames, lang, output_file,
                         max_box=[0, 530, 1280, 720],):
    inpainted_images, bboxes = {}, {}
    img_paths = []

    for region in valid_regions:
        try:
            bbox = json.load(open(os.path.join(image_folders[region[0]], f"bboxes.json")))
        except FileNotFoundError:
            continue
        img_path = os.path.join(image_folders[region[0]], f"inpainted-{lang}.png")
        img_paths.append(img_path)
        inpainted_images[region[0]] = cv2.imread(img_path)
        bboxes[region[0]] = bbox["bbox"]
    # pdb.set_trace()

    idx = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames)

    new_transcript_file = os.path.join(transcript_folder, f"new_transcript_{lang}.json")
    new_transcript = defaultdict(list)
    while video.isOpened():
        ret, frame = video.read()

        if ret == True:
            region = regions_per_frame[idx]
            region_id = region[0]
            if region_id in bboxes:
                inapainted_img = inpainted_images[region_id]
                box = max_box
                xmin, ymin, xmax, ymax = box

                frame[ymin:ymax, xmin:xmax] = inapainted_img[ymin:ymax, xmin:xmax]
            writer.write(frame)
            idx += 1
            pbar.update(1)

        else: break

    writer.release()
    pbar.close()

def build_new_transcript(regions_per_frame, transcript_folder, fps, lang):
    new_transcript = defaultdict(list)
    for frame, region in enumerate(regions_per_frame):
        new_transcript[region[2]].append(frame)

    output = {"transcripts":[], "timestamps": []}
    for text in new_transcript:
        st = str(datetime.timedelta(seconds=int(min(new_transcript[text]) / fps)))
        et = str(datetime.timedelta(seconds=int(max(new_transcript[text]) / fps)))
        output["transcripts"].append(text)
        output["timestamps"].append([st, et])

    json.dump(output, open(os.path.join(transcript_folder, f"new_transcript_{lang}.json"), "w", encoding='utf8'),
              indent=4, ensure_ascii=False)



def build_video(video, valid_regions, regions_per_frame, image_folders, fps,
                width, height, total_frames, lang, output_file):
    inpainted_images, bboxes = {}, {}
    img_paths = []

    for region in valid_regions:
        try:
            bbox = json.load(open(os.path.join(image_folders[region[0]], f"bboxes.json")))
        except FileNotFoundError:
            continue
        img_path = os.path.join(image_folders[region[0]], f"inpainted-{lang}.png")
        img_paths.append(img_path)
        inpainted_images[region[0]] = cv2.imread(img_path)

        bboxes[region[0]] = get_total_bbox(bbox["bbox"], width, height)
    # pdb.set_trace()

    idx = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    pbar = tqdm(total=total_frames)
    while video.isOpened():
        ret, frame = video.read()

        if ret == True:
            region_id = regions_per_frame[idx][0]
            if region_id in bboxes:
                inapainted_img = inpainted_images[region_id]
                box = bboxes[region_id]
                xmin, ymin, xmax, ymax = box

                frame[ymin:ymax, xmin:xmax] = inapainted_img[ymin:ymax, xmin:xmax]
            writer.write(frame)
            idx += 1
            pbar.update(1)

        else: break

    writer.release()
    pbar.close()


def build_video_for_language(transcript_folder, base_video, result_folder, output_folder):
    for lang in langs:
        print("Processing language", langs[lang])
        # Load video and get details
        base_video_name = os.path.splitext(os.path.basename(base_video))[0]
        output_file = os.path.join(output_folder, f"{base_video_name}_{lang}.mp4")
        language_transcript = os.path.join(transcript_folder, f"transcripts_{langs[lang]}.json")

        video = cv2.VideoCapture(base_video)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # List all keyframe result folder in sorted order
        image_folders_base = [(f, int(f.split("_")[1])) for f in os.listdir(result_folder)
                               if os.path.isdir(os.path.join(result_folder, f))]
        image_folders_base = sorted(image_folders_base, key=lambda x: x[1])
        image_folders = [os.path.join(result_folder, f[0]) for f in image_folders_base]

        # Use only those images that have a bounding box modify image folders base accordingly
        image_folders = [x for x in image_folders if os.path.exists(os.path.join(x, "masked.png"))]
        img_ids = [int(os.path.basename(x).split("_")[-1]) for x in image_folders]

        # Load translated trasncript data
        transcript_data = json.load(open(language_transcript, "r"))
        transcripts, timestamps = transcript_data["transcripts"], transcript_data["timestamps"]

        regions_with_text = get_region_with_text(timestamps, transcripts,
                                                 img_ids, fps, image_folders)

        valid_regions, regions_per_frame = get_region_per_frame(regions_with_text,
                                                                total_frames)

        inpainted_images = add_transcripts_to_masked_max_bbox(regions_with_text, image_folders, lang=lang)
        # inpainted_images = add_transcripts_to_masked(regions_with_text, image_folders, lang=lang)

        # Build new video
        # build_video(video, valid_regions, regions_per_frame, image_folders, fps,
        #             width, height, total_frames, lang=lang, output_file=output_file)
        new_transcript = build_new_transcript(regions_per_frame, transcript_folder, fps, lang)
        # pdb.set_trace()
        build_video_max_bbox(video, valid_regions, regions_per_frame, image_folders, fps,
                             width, height, total_frames, lang=lang, output_file=output_file)


if __name__ == "__main__":
    result_folder = "/data/chris/CRAFT-pytorch/result/us_frames_idx"
    base_video = "/data/chris/CRAFT-pytorch/us_air/us_air_force.mp4"
    output_folder = "/data/lucky/video_dubbing/data/us_air"
    transcript_folder = "/data/lucky/video_dubbing/data/us_air/parsed_transcripts"

    # result_folder = "/data/chris/CRAFT-pytorch/result/raw_frames_idx"
    # base_video = "/data/chris/CRAFT-pytorch/ukraine_losses/test.mp4"
    # output_folder = "/data/lucky/video_dubbing/data/ukraine_losses/"
    # transcript_folder = "/data/lucky/video_dubbing/data/ukraine_losses/parsed_transcripts"

    # result_folder = "/data/chris/CRAFT-pytorch/result/obama_frames_idx"
    # base_video = "/data/chris/CRAFT-pytorch/obama_india/obama_india.mp4"
    # output_folder = "/data/lucky/video_dubbing/data/obama_india"
    # transcript_folder = "/data/lucky/video_dubbing/data/obama_india/parsed_transcripts"

    # result_folder = "/data/chris/CRAFT-pytorch/result/russia_frames_idx"
    # base_video = "/data/chris/CRAFT-pytorch/russia_police/russia_police.mp4"
    # output_folder = "/data/lucky/video_dubbing/data/russia_police"
    # transcript_folder = "/data/lucky/video_dubbing/data/russia_police/parsed_transcripts"

    build_video_for_language(transcript_folder, base_video, result_folder, output_folder)

# transformers==4.18.0
