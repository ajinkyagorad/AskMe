import sounddevice as sd
import numpy as np
import queue
import threading
import time
import openai
import soundfile as sf
import io
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from time import time

openai.api_key = <openai-key>
q = queue.Queue()
samplerate = 44100
recording_duration = 50 # timeout (sec)
inactive_time_limit = .5 # when person pauses for this time or more (sec)
recording_blocks = []
dirpath = '/home/pi/Desktop/AskMe/'
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

def text_to_speech(text):
    tts = gTTS(text, lang="en", slow=False)
    tts.save(dirpath+"output.mp3")

    audio_file = AudioSegment.from_file(dirpath+"output.mp3", format="mp3")
    play(audio_file)
def process_audio():
    while True:
        recording_blocks = []
        print('recording...')
        last_active_time = time()
        inactive_time = 0
        start_time = time()
        while True:
            audio_data = q.get()[:,0]
            if np.max(audio_data)<0.01:
               inactive_time = time()-last_active_time
            else:
                last_active_time = time()
            recording_blocks.append(audio_data)
            if inactive_time>inactive_time_limit or len(recording_blocks) * audio_data.shape[0] >= samplerate * recording_duration:
                break
        print('done')
        audio_data_concat = np.concatenate(recording_blocks, axis=0)
        # only proceed if at least 1 second of audio is present else redo
        if time()-start_time<1:
            print('too short')
            continue

        sf.write(dirpath+'input.wav', audio_data_concat, samplerate)
        with open(dirpath+'input.wav', 'rb') as f:
            transcript = openai.Audio.transcribe("whisper-1", f)['text']
            print('input: ',transcript)
        response = openai.Completion.create(engine="text-davinci-002", prompt=transcript, max_tokens=100, n=1, stop=None, temperature=0.7)
        message = response.choices[0].text.strip()
        if transcript:
            print("Detected speech:", transcript)
            print("Response:", message)
            text_to_speech(message)

stream = sd.InputStream(device = 10, callback=audio_callback)
outstream=sd.OutputStream(samplerate=samplerate)
stream.start()
outstream.start()

processing_thread = threading.Thread(target=process_audio)
processing_thread.start()

processing_thread.join()

while True:
    # Keep the main thread running until the user presses the 'q' key
    if input() == 'q':
        break
stream.stop()
stream.close()