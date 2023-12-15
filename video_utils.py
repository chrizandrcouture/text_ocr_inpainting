import os
import pdb
import json
import datetime
from collections import defaultdict
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from tqdm import tqdm


X_PADDING = 10
Y_PADDING = 10
MIN_SENTENCE_LENGTH = 8


def get_min_font_video(regions, boxes, image_folders, fontpath, font_size):
    font_sizes = []
    for (img_id, frame_id, text) in tqdm(regions):
        if len(text.strip()) == 0:
            continue
        text_list = text.split()
        divided_text = divide_list(text_list, len(boxes))

        img_pil = Image.open(os.path.join(image_folders[img_id], "masked.png"))
        draw = ImageDraw.Draw(img_pil)

        adjusted_font_size = get_min_font_for_text(fontpath, font_size, boxes, divided_text, draw, width=None)
        font_sizes.append(adjusted_font_size)

    adjusted_font_size = min(font_sizes)
    return adjusted_font_size


def get_min_font_for_text(fontpath, font_size, boxes, divided_text, draw, width=None):
    font = ImageFont.truetype(fontpath, 32)
    scales = []

    for box, text_list in zip(boxes, divided_text):
        text_str = " ".join(text_list)
        # pdb.set_trace()
        (bbox_x, bbox_y), (xm, ym) = box
        bbox_w = xm - bbox_x
        bbox_h = ym - bbox_y

        text_width, text_height = draw.textsize(text_str, font=font)
        # scale = min(bbox_w / text_width, bbox_h / text_height)
        if width is None:
            scale = bbox_w / text_width
        else:
            scale = width / text_width

        scales.append(scale)

    adjusted_font_size = int(font_size * min(scales))
    return adjusted_font_size


def add_text_to_image(draw, text_str, box, fontpath, adjusted_font_size,
                      font_color, x=None, y=None, width=None, height=None):
    adjusted_font = ImageFont.truetype(fontpath, adjusted_font_size)

    (bbox_x, bbox_y), (xm, ym) = box
    bbox_w = xm - bbox_x
    bbox_h = ym - bbox_y

    x = bbox_x if x is None else x
    y = bbox_y if y is None else y
    width = bbox_w if width is None else width
    height = bbox_h if height is None else height

    new_text_width, new_text_height = draw.textsize(text_str, font=adjusted_font)
    x_pos = x + (width - new_text_width) / 2
    y_pos = y + (height - new_text_height) / 2
    draw.text((x_pos, y_pos), text_str, font=adjusted_font, fill=font_color)


def divide_list(text_list, n):
    quotient, remainder = divmod(len(text_list), n)
    sublists = []
    start = 0

    for i in range(n):
        sublist_size = quotient + 1 if i < remainder else quotient
        sublists.append(text_list[start:start+sublist_size])
        start += sublist_size

    return sublists


def get_total_bbox(bboxes, width=None, height=None, pad=True):
    xmins = [bboxes[x][0][0] for x in bboxes]
    ymins = [bboxes[x][0][1] for x in bboxes]
    xmaxes = [bboxes[x][1][0] for x in bboxes]
    ymaxes = [bboxes[x][1][1] for x in bboxes]

    if pad:
        xmin_all, ymin_all = max(min(xmins) - X_PADDING, 0), max(min(ymins) - Y_PADDING, 0)
        xmax_all, ymax_all = min(max(xmaxes) + X_PADDING, width), min(max(ymaxes) + Y_PADDING, height)
    else:
        try:
            xmin_all, ymin_all = min(xmins), min(ymins)
            xmax_all, ymax_all = max(xmaxes), max(ymaxes)
        except:
            pdb.set_trace()

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


def divide_box(box, num_boxes):
    xmin, ymin, xmax, ymax = box
    start = ymin
    end = ymax
    numbers = np.linspace(start, end, num_boxes+1).astype(int).tolist()
    boxes = [[[xmin, numbers[i]], [xmax, numbers[i+1]]] for i in range(len(numbers)-1)]
    return boxes


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
