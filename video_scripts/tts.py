import os
import argparse
import json
import pdb
import time

from pydub import AudioSegment
import azure.cognitiveservices.speech as speechsdk


parser = argparse.ArgumentParser()
parser.add_argument("--transcripts_folder", type=str, default="transcripts/", help="transcripts file (json)")
parser.add_argument("--output_dir", type=str, default="tts_output/", help="directory to store generated speech")
parser.add_argument("--gender", type=str, default="male", help="glcoud tts sdk language gender choice")
args = vars(parser.parse_args())


language_to_voice_map = {
   "female": {
        "hi": "hi-IN-SwaraNeural",
        "bn": "bn-IN-TanishaaNeural",
        "gu": "gu-IN-DhwaniNeural",
        "kn": "kn-IN-SapnaNeural",
        "ml": "ml-IN-SobhanaNeural",
        "mr": "mr-IN-AarohiNeural",
        "ta": "ta-IN-PallaviNeural",
        "te": "te-IN-ShrutiNeural",
    },
    "male": {
        "hi": "hi-IN-MadhurNeural",
        "bn": "bn-IN-BashkarNeural",
        "gu": "gu-IN-NiranjanNeural",
        "kn": "kn-IN-GaganNeural",
        "ml": "ml-IN-MidhunNeural",
        "mr": "mr-IN-ManoharNeural",
        "ta": "ta-IN-ValluvarNeural",
        "te": "te-IN-MohanNeural"
    }
}

langs = {
        "hi": "hindi",
        "bn": "bengali",
        "gu": "gujarati",
        "mr": "marathi",
        "ta": "tamil",
        "te": "telugu",
        "kn": "kannada",
        "ml": "malayalam"
}


# Creates an instance of a speech config with specified subscription key and service region.
# Replace with your own subscription key and service region (e.g., "westus").
speech_key, service_region = "e344143b679e4260b5bbe777d1992dc8", "eastus"
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)


def create_silent_mp3(duration, output_file):
    silence = AudioSegment.silent(duration=duration * 1000)  # Duration in milliseconds
    silence.export(output_file, format="mp3")


def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.split(':')
    seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return seconds


def translatedspeech(chunk_text, voice_name, chunk_num, output_dir):
    # Set the voice name, refer to https://aka.ms/speech/voices/neural for full list.
    speech_config.speech_synthesis_voice_name = voice_name # "en-US-AriaNeural"
    # Creates a speech synthesizer using a file as audio output.
    output_filepath = os.path.join(output_dir, "chunk"+chunk_num+".mp3")
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_filepath)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    text = chunk_text

    # Synthesizes the received text to speech.
    # The synthesized speech is expected to be heard on the speaker with this line executed.
    result = speech_synthesizer.speak_text_async(text).get()

    # Checks result.
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")


def create_silence(timestamps, output_dir, chunk_num):
    output_filepath = os.path.join(output_dir, "chunk"+chunk_num+".mp3")
    st = timestamp_to_seconds(timestamps[0])
    et = timestamp_to_seconds(timestamps[1])

    duration = et - st
    create_silent_mp3(duration, output_filepath)


if __name__ == "__main__":
    gender = "female"
    output_dir = "/data/lucky/video_dubbing/data/us_air"
    transcript_folder = "/data/lucky/video_dubbing/data/us_air/parsed_transcripts"

    for lang in langs.keys():
        transcript_file = os.path.join(transcript_folder, f"transcripts_{langs[lang]}.json")
        transcripts = json.load(open(transcript_file, "rb"))
        voice_name = language_to_voice_map[gender][lang]
        output_lang_dir = os.path.join(output_dir, "azure_target_clips", lang+"-IN")
        os.makedirs(output_lang_dir, exist_ok=True)
        for i, chunk_text in enumerate(transcripts["transcripts"]):
            if len(chunk_text) == 0:
                # create_silence(transcripts["timestamps"][i], output_lang_dir, str(i+1))
                pass
            else:
                translatedspeech(chunk_text, voice_name, str(i+1), output_lang_dir)
                time.sleep(1)