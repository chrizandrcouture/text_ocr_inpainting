import os
import io
import argparse

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

parser = argparse.ArgumentParser()
parser.add_argument("--audio_dir", type=str, default="audio.wav", help="path to audio file(.wav)")
parser.add_argument("--lang", type=str, default="hi", help="glcoud stt sdk language code")
args = vars(parser.parse_args())


def speechtotext(file_name, lang_code):

    client = speech.SpeechClient()

    print("stt file path",file_name)
    # Loads the audio into memory
    with io.open(file_name, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=lang_code)

    # Detects speech in the audio file
    response = client.recognize(config, audio)
    print(response)
    s = ''
    with open('audiototext.txt', 'a+', encoding='utf8') as f:
        for result in response.results:
            text = str(result.alternatives[0].transcript)
            f.write(text+'. ')
            s += text

    return s


if __name__ == "__main__":
    audio_clips_dir = args["audio_dir"]
    lang_code = args["lang"]
    audio_clips = os.listdir(audio_clips_dir)
    audio_clips = [audio for audio in audio_clips if str(audio).endswith(".wav")]
    for audio_clip in sorted(audio_clips):
        speechtotext(audio_clip, lang_code)

