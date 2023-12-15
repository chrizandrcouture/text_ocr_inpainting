import argparse
import json
import os
import pdb

import cv2
from PIL import Image, ImageDraw
from tqdm import tqdm

from video_utils import get_min_font_for_text, get_total_bbox, get_region_per_frame, \
                        get_region_with_text, divide_box, divide_list, add_text_to_image, \
                        get_min_font_video, build_new_transcript


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


def add_transcripts_to_masked(regions, image_folders, lang):
    font_size = 24
    font_color = (255, 255, 255)
    fontpath = fonts[lang]
    inpainted_images = []

    for (img_id, _, text) in tqdm(regions):
        if len(text.strip()) == 0:
            continue
        bboxes = json.load(open(os.path.join(image_folders[img_id], "bboxes.json")))["bbox"]
        boxes = sorted(list(bboxes.values()), key=lambda x:x[0][1])

        if len(boxes) == 0:
            continue
        max_box = get_total_bbox(bboxes, pad=False)
        max_box_x = max_box[0]
        max_box_width = max_box[2] - max_box[0]

        text_list = text.split()
        divided_text = divide_list(text_list, len(bboxes))

        img_pil = Image.open(os.path.join(image_folders[img_id], "masked.png"))
        draw = ImageDraw.Draw(img_pil)

        adjusted_font_size = get_min_font_for_text(fontpath, font_size, boxes,
                                                   divided_text, draw, width=max_box_width)

        for text_list, box in zip(divided_text, boxes):
            text_str = " ".join(text_list)
            add_text_to_image(draw, text_str, box, fontpath, adjusted_font_size,
                              font_color, x=max_box_x, width=max_box_width)

        output_path = os.path.join(image_folders[img_id], f"inpainted-{lang}.png")
        img_pil.save(output_path)
        inpainted_images.append(output_path)

    return inpainted_images


def add_transcripts_to_masked_max_bbox(regions, image_folders, lang, max_box=[0, 530, 1280, 720], num_lines=2):
    font_size = 24
    font_color = (255, 255, 255)
    fontpath = fonts[lang]

    boxes = divide_box(max_box, num_lines)
    inpainted_images = []
    adjusted_font_size = get_min_font_video(regions, boxes, image_folders, fontpath, font_size)

    for (img_id, _, text) in tqdm(regions):
        if len(text.strip()) == 0:
            continue
        text_list = text.split()
        divided_text = divide_list(text_list, len(boxes))

        img_pil = Image.open(os.path.join(image_folders[img_id], "masked.png"))
        draw = ImageDraw.Draw(img_pil)

        for text_list, box in zip(divided_text, boxes):
            text_str = " ".join(text_list)
            add_text_to_image(draw, text_str, box, fontpath, adjusted_font_size,
                              font_color)

        output_path = os.path.join(image_folders[img_id], f"inpainted-{lang}.png")
        img_pil.save(output_path)
        inpainted_images.append(output_path)

    return inpainted_images


def video_setup(valid_regions, image_folders, lang, width, height, max_bbox=True):
    inpainted_images, bboxes = {}, {}
    img_paths = []
    for region in valid_regions:
        bbox = json.load(open(os.path.join(image_folders[region[0]], f"bboxes.json")))
        if max_bbox:
            bboxes[region[0]] = None
        else:
            if len(bbox["bbox"]) == 0:
                continue
            bboxes[region[0]] =  get_total_bbox(bbox["bbox"], width, height)

        img_path = os.path.join(image_folders[region[0]], f"inpainted-{lang}.png")
        img_paths.append(img_path)
        inpainted_images[region[0]] = cv2.imread(img_path)

    return inpainted_images, bboxes, img_paths


def build_video(video, valid_regions, regions_per_frame, image_folders, fps,
                width, height, total_frames, lang, output_file,
                max_box=[0, 530, 1280, 720]):
    inpainted_images, bboxes, img_paths = video_setup(valid_regions, image_folders, lang,
                                                      width, height, max_bbox=max_box is not None)

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
                box = bboxes[region_id] if max_box is None else max_box
                xmin, ymin, xmax, ymax = box
                frame[ymin:ymax, xmin:xmax] = inapainted_img[ymin:ymax, xmin:xmax]

            writer.write(frame)
            idx += 1
            pbar.update(1)

        else: break

    writer.release()
    pbar.close()


def parse_data(transcript_folder, base_video, result_folder, output_folder, lang):
    # Output file path
    base_video_name = os.path.splitext(os.path.basename(base_video))[0]
    output_file = os.path.join(output_folder, f"{base_video_name}_{lang}.mp4")

    # List all keyframe result folder in sorted order
    image_folders_base = [(f, int(f.split("_")[1])) for f in os.listdir(result_folder)
                           if os.path.isdir(os.path.join(result_folder, f))]
    image_folders_base = sorted(image_folders_base, key=lambda x: x[1])
    image_folders = [os.path.join(result_folder, f[0]) for f in image_folders_base]

    # Use only those images that have a bounding box modify image folders base accordingly
    image_folders = [x for x in image_folders if os.path.exists(os.path.join(x, "masked.png"))]
    img_ids = [int(os.path.basename(x).split("_")[-1]) for x in image_folders]

    # Load translated trasncript data
    language_transcript = os.path.join(transcript_folder, f"transcripts_{langs[lang]}.json")
    transcript_data = json.load(open(language_transcript, "r"))
    transcripts, timestamps = transcript_data["transcripts"], transcript_data["timestamps"]

    return image_folders, img_ids, output_file, transcripts, timestamps


def load_video_details(base_video):
    # Load video and get details
    video = cv2.VideoCapture(base_video)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return video, total_frames, fps, width, height


def build_video_for_language(transcript_folder, base_video, result_folder, output_folder,
                             max_bbox=[0, 530, 1280, 720], regenerate_transcript=True):
    for lang in langs:
        print("Processing language", langs[lang])
        video, total_frames, fps, width, height = load_video_details(base_video)

        image_folders, img_ids, output_file, \
        transcripts, timestamps = parse_data(transcript_folder,
                                             base_video, result_folder,
                                             output_folder, lang)

        regions_with_text = get_region_with_text(timestamps, transcripts,
                                                 img_ids, fps, image_folders)

        valid_regions, regions_per_frame = get_region_per_frame(regions_with_text,
                                                                total_frames)

        if max_bbox is not None:
            add_transcripts_to_masked_max_bbox(regions_with_text, image_folders, lang=lang)
        else:
            add_transcripts_to_masked(regions_with_text, image_folders, lang=lang)

        # Build new video
        if regenerate_transcript:
            build_new_transcript(regions_per_frame, transcript_folder, fps, lang)

        # pdb.set_trace()
        build_video(video, valid_regions, regions_per_frame, image_folders, fps,
                    width, height, total_frames, lang=lang, output_file=output_file,
                    max_box=max_bbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_video', type=str, help='Path to video file')
    parser.add_argument('--output_folder', type=str, help='Path to generate video and new transcripts')
    parser.add_argument('--result_folder', type=str, help='Path to results folder')
    parser.add_argument('--transcript_folder', type=str, help='Path to transcripts folder')
    args = parser.parse_args()

    result_folder = args.result_folder
    base_video = args.base_video
    output_folder = args.output_folder
    transcript_folder = args.transcript_folder

    build_video_for_language(transcript_folder, base_video, result_folder, output_folder,
                            #  max_bbox=None
                             )

    # USAGE:
    # python build_video_refactor.py --result_folder "/data/chris/CRAFT-pytorch/result/us_frames_idx" --base_video "/data/chris/CRAFT-pytorch/frames/us_air/us_air_force.mp4" --output_folder "/data/lucky/video_dubbing/data/us_air" --transcript_folder "/data/lucky/video_dubbing/data/us_air/parsed_transcripts"
    # python build_video_refactor.py --result_folder "/data/chris/CRAFT-pytorch/result/raw_frames_idx" --base_video "/data/chris/CRAFT-pytorch/ukraine_losses/test.mp4" --output_folder "/data/lucky/video_dubbing/data/ukraine_losses/" --transcript_folder "/data/lucky/video_dubbing/data/ukraine_losses/parsed_transcripts"
    # python build_video_refactor.py --result_folder "/data/chris/CRAFT-pytorch/result/obama_frames_idx" --base_video "/data/chris/CRAFT-pytorch/obama_india/obama_india.mp4" --output_folder "/data/lucky/video_dubbing/data/obama_india" --transcript_folder "/data/lucky/video_dubbing/data/obama_india/parsed_transcripts"
    # python build_video_refactor.py --result_folder "/data/chris/CRAFT-pytorch/result/russia_frames_idx" --base_video "/data/chris/CRAFT-pytorch/russia_police/russia_police.mp4" --output_folder "/data/lucky/video_dubbing/data/russia_police" --transcript_folder "/data/lucky/video_dubbing/data/russia_police/parsed_transcripts"

# transformers==4.18.0
