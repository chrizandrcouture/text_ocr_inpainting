import os
import subprocess
import shutil
import argparse
import pdb
import librosa
import numpy as np
import json
import soundfile as sf

import pyrubberband as pyrub
from datetime import datetime, timedelta


parser = argparse.ArgumentParser()
parser.add_argument("--input",type=str, help="input file path (.mp4)")
parser.add_argument("--volume",type=float, default=1, help="background volume between 0 and 1")

target_languages = {
    	"hi-IN": 4, #hindi
    	"bn-IN": 7, #bengali
    	"gu-IN": 3, #gujarati
    	"mr-IN": 6, #marathi
    	"ta-IN": 5, #tamil
    	"te-IN": 2, #telugu
    	"kn-IN": 0, #kannada
    	"ml-IN": 1, #malayalam
}

sts_maps = {
        "hi-IN": "hindi",
        "bn-IN": "bengali",
        "gu-IN": "gujarati",
        "mr-IN": "marathi",
        "ta-IN": "tamil",
        "te-IN": "telugu",
        "kn-IN": "kannada",
        "ml-IN": "malayalam"
}


def time_to_seconds(time_str):
    time_format = "%H:%M:%S.%f"
    time_obj = datetime.strptime(time_str, time_format)
    time_delta = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second, microseconds=time_obj.microsecond)
    return time_delta.total_seconds()


def combine_audio_bg(vocal_file, bg_file, transcripts_file, output_file, SR=16000, volume=1):
    # Load the original audio file
    vocal_audio, sr1 = librosa.load(vocal_file, sr=SR)
    bg_audio, sr2 = librosa.load(bg_file, sr=SR)
    bg_audio *= volume

    timestamps = transcripts_file["timestamps"]
    vocal_duration = librosa.get_duration(vocal_audio, SR)
    target_duration = librosa.get_duration(bg_audio, SR)

    time_stretch_factor =  target_duration / vocal_duration

    # Calculate the segment start and end times in seconds
    segment_start_time = int(time_to_seconds(timestamps[0][0]) * SR)
    segment_end_time = int(time_to_seconds(timestamps[-1][1]) * SR)
    vocal_duration = len(vocal_audio)

    if len(vocal_audio) <= len(bg_audio):
        # pdb.set_trace()
        # Calculate the required padding sizes on the left and right
        padd_len = len(bg_audio) - len(vocal_audio)
        padding_left = int(padd_len / 2)
        padding_right = padd_len - padding_left
        vocal_audio = np.pad(vocal_audio, (padding_left, padding_right), mode='constant')


    elif len(vocal_audio) > len(bg_audio):
        # pdb.set_trace()
        bg_audio = pyrub.time_stretch(bg_audio, SR, time_stretch_factor)


    combined_audio_bg = np.stack((vocal_audio, bg_audio), axis=1)
    combine_audio_bg_mono = librosa.to_mono(combined_audio_bg.T)
    sf.write(output_file, combine_audio_bg_mono, SR, 'PCM_24')
    return time_stretch_factor


def combine_output_audios(out_path):
    dubbed_wavs = [x for x in os.listdir(out_path) if x.endswith("_sst_dub_vc.wav")]
    dubbed_wav_ids = [int(x.split('_')[0].replace("clip", "")) for x in dubbed_wavs]

    dubbed_wavs = [(os.path.join(out_path, wav), id_) for wav, id_ in zip(dubbed_wavs, dubbed_wav_ids)]
    dubbed_wavs = sorted(dubbed_wavs, key=lambda x: x[1])

    dubbed_wav_list = os.path.join(out_path, "dubbed_wav_list.txt")
    stitched_wav = os.path.join(out_path, "sst_dub_stitched.wav")

    with open(dubbed_wav_list, "w") as f:
        f.write("\n".join([f"file '{os.path.abspath(x[0])}'" for x in dubbed_wavs]))

    # Stitch audio files together
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", dubbed_wav_list, "-c", "copy", stitched_wav
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return stitched_wav


def stretch_video_by_factor(video, factor, lang):
    output_file = video.replace(".mp4", f"_{lang}_stretched.mp4")
    command = [
        'ffmpeg', '-y', '-i', video, '-filter:v', f'"setpts={1/factor}*PTS"', '-an',
        # '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-movflags', '+faststart',
        output_file
    ]
    # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.system(" ".join(command))
    # pdb.set_trace()
    return output_file


if __name__ == "__main__":
    args = vars(parser.parse_args())

    input_file = args["input"] #'adani_group_regain_investor/adani_group_regain_investor.mp4'
    volume = args["volume"]
    base_dir = os.path.dirname(input_file)
    basename = os.path.basename(input_file)


    sources_dir = os.path.join(base_dir, "source_components")
    bg_file = os.path.join(sources_dir, os.path.splitext(basename)[0], "accompaniment.wav")

    for lang in target_languages.keys():
        out_path = os.path.join(base_dir, "output", lang)
        lang_video = os.path.join(base_dir, f"{os.path.splitext(basename)[0]}_{lang}.mp4")

        transcripts_folder = os.path.join(base_dir, "parsed_transcripts")
        transcripts_file = json.load(open(os.path.join(transcripts_folder, f"transcripts_{sts_maps[lang]}.json"), "rb"))

        stitched_wav = combine_output_audios(out_path)

        combined_stitched_wav = stitched_wav.replace("sst_dub_stitched.wav", "combined_sst_dub_stitched.wav")
        time_stretch_factor = combine_audio_bg(stitched_wav, bg_file, transcripts_file, combined_stitched_wav, volume=volume)


        if time_stretch_factor < 1:
            stretch_video_file = stretch_video_by_factor(lang_video, time_stretch_factor, lang)

        merged_output_file = input_file.replace(".mp4", "_"+str(target_languages[lang])+".mp4")

        # Replace audio in video
        command = [
                "ffmpeg", "-y", "-i", lang_video if time_stretch_factor >= 1 else stretch_video_file,
                "-i", combined_stitched_wav, "-c:v", "copy", "-c:a", "aac",\
                "-map", "0:v", "-map", "1:a", merged_output_file
        ]
        # pdb.set_trace()
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f'Clip for {lang} created: {merged_output_file}')
