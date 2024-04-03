import numpy as np
import queue
import threading
import time
import cv2
from openai import OpenAI
import whisper
import soundfile as sf
import io
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from time import time
import sounddevice as sd
import os
from playsound import playsound
import simpleaudio
import struct
import boto3
from textblob import TextBlob
import pygame

from PIL import Image
from io import BytesIO
import requests


# AWS access keys (incomplete and are temporary) hint 
aws_access_key_id = ''
aws_secret_access_key = ''
client = OpenAI() #key in environment variable

# AWS region name
region_name = 'eu-central-1'

# Create a client session
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Create a Polly client
polly_client = session.client('polly')
comprehend = session.client('comprehend')
pygame.init()
pygame.mixer.init()


system_message =  ""
q = queue.Queue()
samplerate = 44100
recording_duration = 50 # timeout (sec)
inactive_time_limit = .5 # when person pauses for this time or more (sec)
recording_blocks = []
dirpath = os.path.abspath(os.path.dirname(__file__))

valid_language_codes = ['en-US', 'en-IN', 'es-MX', 'en-ZA', 'tr-TR', 'ru-RU', 'ro-RO', 'pt-PT', 'pl-PL', 'nl-NL', 'it-IT', 'is-IS', 'fr-FR', 'es-ES', 'de-DE', 'yue-CN', 'ko-KR', 'en-NZ', 'en-GB-WLS', 'hi-IN', 'arb', 'cy-GB', 'cmn-CN', 'da-DK', 'en-AU', 'pt-BR', 'nb-NO', 'sv-SE', 'ja-JP', 'es-US', 'ca-ES', 'fr-CA', 'en-GB', 'de-AT']

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())
def text_to_speech_aws(text):
    response = comprehend.detect_dominant_language(Text=text)

    # extract language code
    language_code = response['Languages'][0]['LanguageCode']
    language_code = language_code+'-'+language_code.upper()
    

    if language_code not in valid_language_codes:
        language_code = valid_language_codes[0] 

    # Generate an MP3 file using Polly
    print('detected as:', language_code)
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna',
        LanguageCode=language_code
    )
    ofile = os.path.join(dirpath, 'output.mp3')
    # Save the MP3 file to disk
    pygame.mixer.music.unload() 

    try:
        with open(ofile, 'wb') as file:
            file.write(response['AudioStream'].read())
    except Exception as e:
        print(f"Error saving file: {e}")

    print('##')
    play_mp3(ofile)     
def text_to_speech_aws2(text):
    response = comprehend.detect_dominant_language(Text=text)

    # extract language code
    language_code = response['Languages'][0]['LanguageCode']
    language_code = language_code+'-'+language_code.upper()
    

    if language_code not in valid_language_codes:
        language_code = valid_language_codes[0] 

    # Generate an MP3 file using Polly
    print('detected as:', language_code)
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna',
        LanguageCode=language_code
    )
    ofile = os.path.join(dirpath, 'output.mp3')
    # Save the MP3 file to disk
    with open(ofile, 'wb') as file:
        file.write(response['AudioStream'].read())
    ofile = ofile.replace('\\', '\\\\')
    print(ofile)
    sound = AudioSegment.from_mp3(ofile)
    sound.export('output.wav', format="wav")
    obj = simpleaudio.WaveObject.from_wave_file('output.wav')
    pobj = obj.play()
    pobj.wait_done()

def play_mp3(file_path, speed=1.0):
    
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
  

def text_to_speech(text):
    tts = gTTS(text, lang="en", slow=False)
    ofile = os.path.join(dirpath, 'output.mp3')
    pygame.mixer.music.unload() 

    try:
        tts.save(ofile)
    except Exception as e:
        print(f"Error saving file: {e}")

    print('##')
    play_mp3(ofile)


def draw_image(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    print('submitted request with prompt:',prompt,' ... waiting')
    # Adjusted to match the new response structure
    image_url = response.data[0].url
    response = requests.get(image_url)
    print('response at: ', image_url)
    
    # Open the response content as an image using Pillow
    image = Image.open(BytesIO(response.content))
    # Convert the image to an array format that OpenCV can work with
    image_array = np.array(image)
    # OpenCV expects colors in BGR format, whereas Pillow provides them in RGB.
    # Convert from RGB to BGR format
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Generated Image', image_array)
    cv2.waitKey(5000)  # Display the window for 5 seconds
    cv2.destroyAllWindows()

def text_to_speech2(text):
    tts = gTTS(text, lang="en", slow=False)
    #fp = io.BytesIO()
    ofile = os.path.join(dirpath, 'output.mp3')
    tts.save(ofile)
    print(ofile)
    sound = AudioSegment.from_mp3(ofile)
    sound.export('output.wav', format="wav")
    obj = simpleaudio.WaveObject.from_wave_file('output.wav')
    pobj = obj.play()
    pobj.wait_done()
    #print('filepath: ',os.path.join(dirpath, 'output.mp3')) 
    
    #audio_file = AudioSegment.from_file('output.mp3', format="mp3")
    #play(audio_file)
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
        # only proceed if at least 1 second of audio is present and there is at least 50% audio else redo
        if time()-start_time<1:
            print('too short')
            continue
        val=np.sum(audio_data_concat>0.005)/len(audio_data_concat)
        if val<.05:
            print('too little audio :', val)
            continue
        try:
            sf.write(os.path.join(dirpath, 'input.wav'), audio_data_concat, samplerate)
            print('saved.')
        except Exception as e:
            print(f"Error saving audio: {e}")

        
        # Correctly use the OpenAI API for transcribing an audio file
        with open(os.path.join(dirpath, 'input.wav'), 'rb') as audio_file:
            #print('Trying to transcribe...')
            transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
            )
            print('Input:', transcription.text)
            transcript = transcription.text

        
        # Checks if the first word of the transcript is "DRAW", which triggers the drawing function.
        if transcript.split(' ')[0].upper() == 'DRAW':
            draw_image(transcript)

        # Checks if the first word of the transcript is "QUIT", which exits the program.
        elif transcript.strip().upper().split(' ')[0] == 'QUIT':
            exit()

        # Checks if the first word of the transcript is "SYSTEM", which updates the system role dynamically.
        elif transcript.upper().startswith('SYSTEM'):
            # Extracts the new role description after "SYSTEM"
            new_role_description = ' '.join(transcript.split(' ')[1:])
            system_message = f"You are now a, {new_role_description}. Restrict responses to few sentences, unless asked."
            print(f"System role updated to: {new_role_description}")

        # For any other transcript, proceeds with generating a chat completion.
        else:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[
                    # The system role can be dynamically updated based on the previous "SYSTEM" command.
                    # This message sets the context for the AI's personality and response restrictions.
                    {"role": "system", "content": system_message},  
                    {"role": "user", "content": transcript}
                ], 
                max_tokens=100, 
                temperature=0.5  # Adjusted for a balance between determinism and creativity.
            )

            # Extracts and prints the AI-generated message.
            message = response.choices[0].message.content
            print("Response:", message)
            text_to_speech_aws(message)


stream = sd.InputStream(device = 0, callback=audio_callback)
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
