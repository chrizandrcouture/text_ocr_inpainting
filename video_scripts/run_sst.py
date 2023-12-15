from TTS.api import TTS
import pdb
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import moviepy.editor as mpe
from spleeter.separator import Separator
import torch


class Dubber:
    def __init__(self, model_name, gpu, workdir) -> None:
        self.model = TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        self.workdir = workdir
        self.separator = Separator("spleeter:2stems")

    def extract_audio(self, video_path, output_path):
        command = f'ffmpeg -y -i "{video_path}" -ac 2 -f wav "{output_path}"'
        os.system(command)
        return output_path

    def change_voice(self, input_wav, reference_audio, output_path):
        with torch.no_grad():
            self.model.voice_conversion_to_file(source_wav=input_wav, target_wav=reference_audio, file_path=output_path)
        return output_path

    def combine_vocals_bg(self, vocals, background, output_path):
        audio_background = mpe.AudioFileClip(background)
        audio_vocals = mpe.AudioFileClip(vocals)
        combined = mpe.CompositeAudioClip([audio_vocals, audio_background])
        combined.write_audiofile(output_path, codec="pcm_s16le", fps=44100)
        return output_path

    def split_source(self, audio_file, video_dir):
        name = os.path.splitext(os.path.basename(audio_file))[0]
        self.separator.separate_to_file(audio_file, video_dir)
        vocal = os.path.join(video_dir, name, "vocals.wav")
        background = os.path.join(video_dir, name, "accompaniment.wav")
        return vocal, background

    def dub(self, original_video, dubbed_video):
        name = os.path.splitext(os.path.basename(original_video))[0]
        dubname = os.path.splitext(os.path.basename(dubbed_video))[0]
        video_dir = os.path.join(self.workdir, dubname)
        os.makedirs(video_dir, exist_ok=True)

        original_wav = self.extract_audio(original_video, os.path.join(video_dir, f"original-{name}.wav"))
        dubbed_wav = self.extract_audio(dubbed_video, os.path.join(video_dir, f"dubbed-{name}.wav"))

        vc_wav = self.change_voice(dubbed_wav, original_wav, os.path.join(video_dir, f"vc-{name}.wav"))
        vocals, background = self.split_source(original_wav, video_dir)

        final_wav = self.combine_vocals_bg(vc_wav, background, os.path.join(video_dir, "final.wav"))


def convert_to_wavs(mp3_files):
    wav_paths = []
    for mp3 in mp3_files:
        wav_path = mp3.replace(".mp3", ".wav")
        command = f'ffmpeg -y -i "{mp3}" "{wav_path}"'
        os.system(command)
        wav_paths.append(wav_path)
    return wav_paths


def extract_video_sources(sources_dir, input_basename, input_video):
    os.makedirs(sources_dir, exist_ok=True)
    src_wav = os.path.join(sources_dir, input_basename.replace(".mp4", ".wav"))
    dubber.extract_audio(input_video, src_wav)
    dubber.split_source(src_wav, sources_dir)


def extract_clip_sources(dubber, input_basedir, lang, source_clips):
    sources_dir_lang = os.path.join(input_basedir, "source_components", lang)
    os.makedirs(sources_dir_lang, exist_ok=True)

    vocal_srcs = []
    for src_clip in source_clips:
        vocal, bg = dubber.split_source(src_clip, sources_dir_lang)
        vocal_srcs.append(vocal)

    return vocal_srcs


def get_azure_clips_sorted(input_basedir, lang, source_clips):
    dubbed_clips_dir = os.path.join(input_basedir, "azure_target_clips", lang)
    dubbed_mp3s = [x for x in os.listdir(dubbed_clips_dir) if x.endswith(".mp3")]
    dubbed_mp3_ids = [int(x.replace(".mp3", "").replace("chunk", "")) for x in dubbed_mp3s]
    dubbed_mp3s = sorted([(dubbed_mp3, dubbed_mp3_id) for dubbed_mp3, dubbed_mp3_id in zip(dubbed_mp3s, dubbed_mp3_ids)], key=lambda x: x[1])
    dubbed_vocal_mp3s = [os.path.join(dubbed_clips_dir, x[0]) for x in dubbed_mp3s]
    if len(dubbed_vocal_mp3s) > len(source_clips):
        dubbed_vocal_mp3s = dubbed_vocal_mp3s[0 : len(source_clips)]
    return dubbed_vocal_mp3s


target_languages = {
    "hi-IN": 4,  # hindi
    "bn-IN": 7,  # bengali
    "gu-IN": 3,  # gujarati
    "mr-IN": 6,  # marathi
    "ta-IN": 5,  # tamil
    "te-IN": 2,  # telugu
    "kn-IN": 0,  # kannada
    "ml-IN": 1,  # malayalam
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path of input video (mp4)")
    parser.add_argument("--target_voice", type=str, help="path of target voice")
    args = vars(parser.parse_args())
    input_video = args["input"]
    target_voice = args["target_voice"]

    input_basedir = os.path.dirname(input_video)
    input_basename = os.path.basename(input_video)

    dubber = Dubber(model_name="voice_conversion_models/multilingual/vctk/freevc24", gpu=True, workdir="output/")

    # Extract BG + vocals from source
    sources_dir = os.path.join(input_basedir, "source_components")
    extract_video_sources(sources_dir, input_basename, input_video)

    for lang in target_languages.keys():
        input_clips_dir = os.path.join(input_basedir, "source_clips", lang)

        # Get list of clips in sorted IDs
        clips = [x for x in os.listdir(input_clips_dir) if x.endswith(".mp4")]
        clip_ids = [int(x.replace(".mp4", "").replace("clip", "")) for x in clips]
        clips = sorted([(clip, clip_id) for clip, clip_id in zip(clips, clip_ids)], key=lambda x: x[1])
        source_clips = [os.path.join(input_clips_dir, x[0]) for x in clips]
        source_wavs = [x.replace(".mp4", ".wav") for x in source_clips]

        # Extract vocals for each clip
        vocal_srcs = extract_clip_sources(dubber, input_basedir, lang, source_clips)

        # Sort azure audio for each clip by IDs and use only those for which sources avaiable
        dubbed_vocal_mp3s = get_azure_clips_sorted(input_basedir, lang, source_clips)

        # Convert MP3s to WAVs
        dubbed_vocal_wavs = convert_to_wavs(dubbed_vocal_mp3s)

        output_dir = os.path.join(input_basedir, "output", lang)
        os.makedirs(output_dir, exist_ok=True)

        print(source_clips)
        for src_clip, vocal_src, dubbed_vc_wav in zip(source_clips, vocal_srcs, dubbed_vocal_wavs):
            # pdb.set_trace()
            src_name = os.path.splitext(os.path.basename(src_clip))[0]
            sst_dub_vc_path = os.path.join(output_dir, src_name + "_sst_dub_vc.wav")
            if len(target_voice) > 0:
                sst_dub_vocal = dubber.change_voice(dubbed_vc_wav, target_voice, sst_dub_vc_path)
            else:
                sst_dub_vocal = dubber.change_voice(dubbed_vc_wav, vocal_src, sst_dub_vc_path)
            print("done: ", sst_dub_vc_path)
