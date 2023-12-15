import os
import subprocess
import shutil
import argparse
import pdb
import librosa
import numpy as np
import json
import soundfile as sf

from tqdm import tqdm
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


def time_to_seconds(time_str, milliseconds=False):
    if milliseconds:
        time_format = "%H:%M:%S.%f"
    else:
        time_format = "%H:%M:%S"
    time_obj = datetime.strptime(time_str, time_format)
    time_delta = timedelta(hours=time_obj.hour, minutes=time_obj.minute, seconds=time_obj.second, microseconds=time_obj.microsecond)
    return time_delta.total_seconds()


def get_duration(input_file):
    audio, sr = librosa.load(input_file)
    duration = librosa.get_duration(audio, sr=sr)
    return duration


def combine_audio_bg(vocal_file, bg_file, output_file, SR=16000, volume=1):
    # Load the original audio file
    vocal_audio, sr1 = librosa.load(vocal_file, sr=SR)
    bg_audio, sr2 = librosa.load(bg_file, sr=SR)
    bg_audio *= volume

    vocal_duration = librosa.get_duration(vocal_audio, SR)
    target_duration = librosa.get_duration(bg_audio, SR)

    time_stretch_factor =  target_duration / vocal_duration

    # Calculate the segment start and end times in seconds
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


def stretch_audio(audio_file, stretch_factor, output_file):
    audio, sr = librosa.load(audio_file)
    new_audio = pyrub.time_stretch(audio, sr, stretch_factor)
    sf.write(output_file, new_audio, sr, 'PCM_24')


def stitch_wavs(out_path, wavs, output_file):
    wav_list = os.path.join(out_path, "wav_list.txt")

    with open(wav_list, "w") as f:
        f.write("\n".join([f"file '{os.path.abspath(x[0])}'" for x in wavs]))

    # Stitch audio files together
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", wav_list, "-c", "copy", output_file
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output_file


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


def stretch_video_and_audio_by_factor(video, factor):
    output_file = video.replace(".mp4", f"_stretched.mp4")
    command = [
        'ffmpeg', '-y', '-i', f'"{video}"', '-filter_complex',
        f'"[0:v]setpts={1/factor}*PTS[v];[0:a]atempo={factor}[a]"',
        '-map', '[v]', '-map', '[a]', f'"{output_file}"'
    ]
    # pdb.set_trace()
    os.system(" ".join(command))
    return output_file


def get_dubbed_files(audio_folder, sort=True):
    dubbed_wavs = [x for x in os.listdir(audio_folder) if x.endswith("_sst_dub_vc.wav") and
                   not x.endswith("aligned_sst_dub_vc.wav")]
    dubbed_wav_ids = [int(x.split('_')[0].replace("clip", "")) for x in dubbed_wavs]

    dubbed_wavs = [(os.path.join(audio_folder, wav), id_) for wav, id_ in zip(dubbed_wavs, dubbed_wav_ids)]
    if sort:
        dubbed_wavs = sorted(dubbed_wavs, key=lambda x: x[1])

    return dubbed_wavs


def align_and_combine(transcripts, dubbed_wavs):
    # newtexts, newtimestamps = new_transcripts["transcripts"], new_transcripts["timestamps"]
    texts, timestamps = transcripts["transcripts"], transcripts["timestamps"]
    new_wavs = []
    target_durations = [time_to_seconds(x[1]) - time_to_seconds(x[0]) for x in timestamps]
    current_durations = [get_duration(wav[0]) for wav in dubbed_wavs]
    speedup = [x/y for x, y in zip(current_durations, target_durations)]
    most_speedup = max(1, max(speedup))
    new_durations = [x * most_speedup for x in target_durations]

    print("aligning audio files")
    for i, (old, new, wav) in enumerate(tqdm(zip(current_durations, new_durations, dubbed_wavs), total=len(dubbed_wavs))):
        vocal_audio, sr = librosa.load(wav[0])
        aligned_path = wav[0].replace("sst_dub_vc.wav", "aligned_sst_dub_vc.wav")
        curr_len, new_len = int(old * sr), int(new * sr)

        assert new_len >= curr_len

        padd_len = new_len - curr_len
        padding_left = int(padd_len / 2)
        padding_right = padd_len - padding_left
        vocal_audio = np.pad(vocal_audio, (padding_left, padding_right), mode='constant')
        sf.write(aligned_path, vocal_audio, sr, 'PCM_24')
        new_wavs.append((aligned_path, i))

    return most_speedup, new_wavs


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
        new_transcripts_file = json.load(open(os.path.join(transcripts_folder, f"new_transcript_{lang}.json"), "rb"))

        dubbed_wavs = get_dubbed_files(out_path, sort=True)
        # pdb.set_trace()
        time_stretch_factor, stitched_aligned_wav = align_and_combine(new_transcripts_file, dubbed_wavs)

        # stitched_wav = os.path.join(out_path, "sst_dub_stitched.wav")
        # stitch_wavs(out_path, stitched_aligned_wav, stitched_wav)

        # print(bg_file)
        # combined_stitched_wav = stitched_wav.replace("sst_dub_stitched.wav", "combined_sst_dub_stitched.wav")
        # combine_audio_bg(stitched_wav, bg_file, combined_stitched_wav, volume=volume)


        # if time_stretch_factor > 1:
        #     stretch_video_file = stretch_video_by_factor(lang_video, 1/time_stretch_factor, lang)

        merged_output_file = input_file.replace(".mp4", "_"+str(target_languages[lang])+".mp4")

        # # Replace audio in video
        # command = [
        #         "ffmpeg", "-y", "-i", lang_video if time_stretch_factor <= 1 else stretch_video_file,
        #         "-i", combined_stitched_wav, "-c:v", "copy", "-c:a", "aac",\
        #         "-map", "0:v", "-map", "1:a", merged_output_file
        # ]
        # # pdb.set_trace()
        # subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        new_output_file = stretch_video_and_audio_by_factor(merged_output_file, (1+time_stretch_factor)/2)
        print(f'Clip for {lang} created: {merged_output_file} and {new_output_file}')
