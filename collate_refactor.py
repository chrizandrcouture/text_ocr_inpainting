import argparse
import datetime
import json
import os
import pdb

import cv2


def get_images_timestamps(result_folder, video_path):
    # Get image folders and sort them based on their IDs
    folders = [f for f in os.listdir(result_folder)
               if os.path.isdir(os.path.join(result_folder, f))]
    sorted_folders = sorted(folders, key=lambda folder: int(folder.split("_")[1]))
    frame_ids = [int(folder.split("_")[1]) for folder in sorted_folders]

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    start_time = str(datetime.timedelta(seconds=0))
    time_stamps = [start_time] + [str(datetime.timedelta(seconds=int(frame_id / fps))) for frame_id in frame_ids]

    return sorted_folders, time_stamps


def collate_results(image_folders, time_stamps, result_folder):
    result = {}

    for i in range(len(time_stamps) - 1):
        start_time = time_stamps[i]
        end_time = time_stamps[i + 1]
        key = f"{start_time}-{end_time}"
        ocr_output = json.load(open(os.path.join(result_folder, image_folders[i], "ocr_result.json")))

        imgs = list(ocr_output.keys())
        img_ids = [int(img.split("_")[1].replace(".png", "")) for img in imgs]
        sorted_imgs = sorted(list(zip(imgs, img_ids)), key=lambda x:x[1])

        text = [ocr_output[img] for img, id_ in sorted_imgs]
        final_text = "\n".join(text)

        result[key] = final_text

    output_filename = os.path.join(result_folder, "collated.json")
    with open(output_filename, 'w') as file:
        json.dump(result, file, indent=4)

    return output_filename


def combine_consecutive_transcripts(corrected_transcript_file):
    with open(corrected_transcript_file, 'r') as file:
        result = json.load(file)
    # Combine consecutive timestamps with the same text
    combined_data = {}
    current_timestamp = None
    current_text = None

    # Sort the keys based on start times in the timestamp
    sorted_keys = sorted(result.keys(), key=lambda x: datetime.datetime.strptime(x.split('-')[0], '%H:%M:%S'))
    # start_times = list(map(lambda x: datetime.datetime.strptime(x.split('-')[0], '%H:%M:%S'), result.keys()))

    for timestamp in sorted_keys:
        text = result[timestamp]
        if current_text is None:
            current_text = text
            current_timestamp = timestamp
        elif text == current_text:
            current_timestamp = current_timestamp.split('-')[0] + '-' + timestamp.split('-')[1]
        else:
            combined_data[current_timestamp] = current_text
            current_text = text
            current_timestamp = timestamp

    # Add the last combined timestamp
    if current_text is not None:
        combined_data[current_timestamp] = current_text

    # Save the combined data to a new JSON file
    with open(os.path.join(result_folder, "combined_transcript.json"), 'w') as file:
        json.dump(combined_data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--corrected_transcript', type=str, help='Path to corrected transcripts file')
    parser.add_argument('--result_folder', type=str, help='Path to results folder')
    args = vars(parser.parse_args())

    result_folder = args.result_folder
    video_path = args.video
    corrected_transcript_file = args.corrected_transcript

    image_folders, time_stamps = get_images_timestamps(result_folder, video_path)
    collated_file = collate_results(image_folders, time_stamps, result_folder)

    combined_file = combine_consecutive_transcripts(corrected_transcript_file)

    # USAGE:
    # python collate.py --video_path "/data/chris/CRAFT-pytorch/us_air/us_air_force.mp4" --corrected_transcript_file "/data/chris/CRAFT-pytorch/result/us_frames_idx/corrected.json" --result_folder /data/chris/CRAFT-pytorch/result/us_frames_idx/
