import pafy
from typing import List
import requests
import m3u8
import cv2
import threading
import shutil
import os
import moviepy.editor as editor
import moviepy.audio.AudioClip as AudioClip
import numpy as np
import json
from os.path import join, dirname
import threading
import wave
import contextlib
import sys
import collections
import time
import re
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader
from google.cloud import speech_v1p1beta1 as speech
import pyaudio


###################################################################################################
# my code
###################################################################################################
complete_history = [0] * 3600 * 100
# multi-thread download stream video
class download_thread (threading.Thread):
    def __init__(self, url:str, index:int):
        threading.Thread.__init__(self)
        self.url = url
        self.index = index
        global thread_pool
        thread_pool.append(self)
    def run(self):
        r = requests.get(self.url)
        f = open("cache/video{}.ts".format(self.index), "wb")
        f.write(r.content)
        f.close()

        # video = editor.VideoFileClip("cache/video{}.ts".format(self.index))
        # audio = video.audio
        # buffer = audio.reader.buffer
        # buffer = buffer.mean(axis=1)
        # sampleRate = audio.fps
        # from scipy.io.wavfile import write
        # scaled = np.int16(buffer *  32767)
        # write("cache/audio{}.wav".format(self.index), sampleRate, scaled)
        # import librosa    
        # y, s = librosa.load("cache/audio{}.wav".format(self.index), sr=8000)
        # write_nparray_to_wav(y, "cache/audio{} sample8k.wav".format(self.index), 8000)
        
        global complete_history
        complete_history[self.index] = 1
        print("video clip {} download complete".format(self.index))




# multi-thread play video
class play_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        #start the video
        video_list = os.listdir("cache")
        index = 0
        print("start playing")
        while 1:
            cap = cv2.VideoCapture("cache/video{}.ts".format(index))
            video_list = os.listdir("cache")
            index += 1
            while (True):
                ret,frame = cap.read()
                if not ret:
                    break
                """
                your code here
                """
                cv2.imshow('frame',frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break    

            cap.release()
        print("playing end")

class audio_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        global thread_pool
        thread_pool.append(self)
    def run(self):

        index = 1
        chunksize = 1
        time.sleep(chunksize)

        print("start playing")
        while 1:
            print("processing index={}".format(index))

            def check_complete(index, len):
                global complete_history
                for i in range(index, index+len):
                    if complete_history[i] == 0:
                        return False
                return True
            if not check_complete(index, chunksize):
                print("index={} download not finished".format(index))
                if check_complete(index+chunksize, chunksize):
                    index += chunksize
                    continue
                else:
                    time.sleep(chunksize)
                    continue

            continue_flag = 0
            for i in range(0, chunksize):
                video = editor.VideoFileClip("cache/video{}.ts".format(index+i))
                audio = video.audio
                buffer = audio.reader.buffer
                buffer = buffer.mean(axis=1)
                sampleRate = audio.fps
                from scipy.io.wavfile import write
                scaled = np.int16(buffer *  32767)
                write("cache/audio{}.wav".format(index+i), sampleRate, scaled)

                video.reader.close()
                video.audio.reader.close_proc()


            buffer = ""
            buffer_list = []
            with contextlib.closing(wave.open("cache/audio{}.wav".format(index), 'rb')) as wf:
                pcm_data = wf.readframes(wf.getnframes())
                buffer = pcm_data
                buffer_list.append(buffer)
            for i in range(1, chunksize):
                with contextlib.closing(wave.open("cache/audio{}.wav".format(index+i), 'rb')) as wf:
                    pcm_data = wf.readframes(wf.getnframes())
                    # buffer += pcm_data
                    buffer_list.append(pcm_data)
            

            global streaming_config, speech, client
            requests = (speech.types.StreamingRecognizeRequest(audio_content=buffer) for chunk in buffer_list)
            responses = client.streaming_recognize(streaming_config, requests)
            num_chars_printed = 0

            for response in responses:
                if not response.results:
                    continue

                # The `results` list is consecutive. For streaming, we only care about
                # the first result being considered, since once it's `is_final`, it
                # moves on to considering the next utterance.
                result = response.results[0]
                if not result.alternatives:
                    continue

                # Display the transcription of the top alternative.
                transcript = result.alternatives[0].transcript

                # Display interim results, but with a carriage return at the end of the
                # line, so subsequent lines will overwrite them.
                #
                # If the previous result was longer than this one, we need to print
                # some extra spaces to overwrite the previous result
                overwrite_chars = ' ' * (num_chars_printed - len(transcript))

                if not result.is_final:
                    sys.stdout.write(transcript + overwrite_chars + '\r')
                    sys.stdout.flush()

                    num_chars_printed = len(transcript)

                else:
                    print(transcript + overwrite_chars)

                    # Exit recognition if any of the transcribed phrases could be
                    # one of our keywords.
                    if re.search(r'\b(exit|quit)\b', transcript, re.I):
                        print('Exiting..')
                        break
   
            index += chunksize

                

        print("playing end")


# array values should be [-1,1]
def write_nparray_to_wav(array, filename, sampleRate):
    from scipy.io.wavfile import write
    scaled = np.int16(array *  32767)
    write(filename, sampleRate, scaled)


def clear_cache():
    shutil.rmtree('cache/*', ignore_errors=True)
    os.makedirs("cache")



if __name__ == "__main__" and 1:
    import signal
    def signal_handler(sig, frame):
        print('clearing cache')
        shutil.rmtree('cache/*', ignore_errors=True)
        for thread in thread_pool:
            thread.stopped = True
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # os.makedirs("cache")
    # clear_cache()

    thread_pool = []

    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='ja-JP',
        max_alternatives=1)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)


    stream = pafy.new("https://www.youtube.com/watch?v=MFLcCM5Qz_o")
    videofile_history = []

    count = 0
    thread = audio_thread()
    thread.start()
    
    while 1:
        play = stream.getbest()
        response = requests.get(play.url)

        m3u8text:str = response.text
        m3u8obj = m3u8.loads(m3u8text)

        for videofile in m3u8obj.files:
            if videofile in videofile_history:
                continue
            else:
                videofile_history.append(videofile)
            thread_new = download_thread(url=videofile, index=count)
            thread_new.start()
            count += 1
            
if __name__ == "__main__":
    thread = audio_thread()
    thread.start()

