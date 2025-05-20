from os import system
from gpt4all import GPT4All
import sys
import whisper
import threading
import pyaudio
import wave
import math
import threading

isDarwin = False
if sys.platform == 'darwin':
    isDarwin = True
    import tensorflow as tf

audio_file_name = 'recordedFile.wav'
chat_model = GPT4All("/Users/ericagostinho/Library/Application Support/nomic.ai/GPT4All/gpt4all-falcon-newbpe-q4_0.gguf", allow_download=False)
recording = True
audio_model = whisper.load_model('base')
should_talk = True

t1 = None

def respond(text):
    if sys.platform == 'darwin':
        ALLOWED_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?!-_$:+-/ ")
        clean_text = ''.join(c for c in text if c in ALLOWED_CHARS)
        system(f"say '{clean_text}'")
    else:
        engine.say(text)
        engine.runAndWait()

def transcribe():
    output = audio_model.transcribe(audio_file_name, fp16=False)
    if output and output['text']:
        print("You said:", output['text'])
        return output['text'].lower()
    print("Could not understand audio. Please try again.")
    return None


def waitToStartTalking():
    input()

def recordAudio():
    global recording
    global audio_file_name

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 1
    audio = pyaudio.PyAudio()

    print("Recording...")

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index = 0,
                    frames_per_buffer=CHUNK)
    Recordframes = []

    
    while recording:
        for i in range(0, math.ceil(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)

    print("Stopped recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile = wave.open(audio_file_name, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    for frame in Recordframes:
        waveFile.writeframes(frame)
    waveFile.close()

def stopRecording():
    global recording
    input()
    recording = False

def startThread():
    global t1
    t1 = threading.Thread(target=stopRecording)
    t1.start()

def stopThread():
    global t1
    t1.join()

should_run = True
def main():
    global should_run
    global recording
    with chat_model.chat_session():
        respond('Hi there! Press enter to talk and press it again to stop talking')

        while should_run:
            recording = True
            waitToStartTalking()
            startThread()
            recordAudio()
            stopThread()
            transcribed = transcribe()
            
            if not transcribed: 
                continue

            if 'bye' in transcribed: 
                should_run = False
                break

            output = chat_model.generate(transcribed)

            print("Output: ", output)
            if should_talk:
                respond(output)
        
        respond('Good bye!')



if __name__ == "__main__":
    if isDarwin and tf.config.list_physical_devices('GPU'):
        with tf.device("/GPU:0"):
            main()
    else:
        main()