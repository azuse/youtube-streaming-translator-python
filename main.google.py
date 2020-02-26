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
from multiprocessing import Process,Array


###################################################################################################
# my code
###################################################################################################
complete_history = Array('i', [0]*3600*24)

# multi-thread download stream video
class download_thread (threading.Thread):
    def __init__(self, url:str, index:int, complete_history):
        threading.Thread.__init__(self)
        self.url = url
        self.index = index
        self.complete_history = complete_history
        

    def run(self):
        proxy = {"http":"http://127.0.0.1:7890","https":"http://127.0.0.1:7890"}
        try:
            r = requests.get(self.url, proxies=proxy)
        except:
            print("download fail index={}".format(self.index))
            try:
                r = requests.get(self.url, proxies=proxy)
            except:
                print("download fail index={}".format(self.index))
                return

        
        f = open("cache/video{}.ts".format(self.index), "wb")
        f.write(r.content)
        f.close()

        self.complete_history[self.index] = 1
        # print("video clip {} download complete".format(self.index))


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

                cv2.imshow('frame',frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break    

            cap.release()
        print("playing end")


multi_thread_buffer_list = []
class multi_thread_read_buffer(threading.Thread):
    def __init__(self, multi_thread_buffer_list, index, i):
        self.multi_thread_buffer_list = multi_thread_buffer_list
        self.index = index
        self.i = i
        threading.Thread.__init__(self)
    def run(self):
        video = editor.VideoFileClip("cache/video{}.ts".format(self.index + self.i))
        print("read {}".format(self.index + self.i))
        audio = video.audio
        buffer_tmp = audio.reader.buffer
        buffer_tmp = buffer_tmp.mean(axis=1)
        sampleRate = audio.fps
        self.multi_thread_buffer_list[self.i] = buffer_tmp
        video.reader.close()
        video.audio.reader.close_proc()

# multi thread extract audio and speech2text and translation
class audio_thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):

        index = 1
        chunksize = 6
        timesleep = 1
        time.sleep(timesleep)

        print("start playing")
        while 1:
            print("processing index={}".format(index))

            continue_flag = 0
            buffer = np.array([])
            i = 0

            global multi_thread_buffer_list
            multi_thread_buffer_list = [0] * chunksize
            threadpool = []
            while i < chunksize:
                global complete_history
                if complete_history[index + i] != 1:
                    continue

                read_thread = multi_thread_read_buffer(multi_thread_buffer_list, index, i)
                read_thread.start()
                threadpool.append(read_thread)
                
                i+=1

            for i,thread in enumerate(threadpool):
                thread.join()
                buffer = np.concatenate([buffer, multi_thread_buffer_list[i]])
            
            
            

            from scipy.io.wavfile import write
            scaled = np.int16(buffer *  32767)
            write("cache/audio{}.wav".format(index), 44100, scaled)
            print("wav written")


            buffer = ""
            buffer_list = []
            with contextlib.closing(wave.open("cache/audio{}.wav".format(index), 'rb')) as wf:
                pcm_data = wf.readframes(wf.getnframes())
                buffer = pcm_data
                buffer_list.append(buffer)
            
            

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

            # 调谷歌翻译 TODO
            # 命令行显示优化 TODO


            index += chunksize

                

        print("playing end")


# array values should be [-1,1]
def write_nparray_to_wav(array, filename, sampleRate):
    from scipy.io.wavfile import write
    scaled = np.int16(array *  32767)
    write(filename, sampleRate, scaled)


def clear_cache():
    try:
        shutil.rmtree('cache', ignore_errors=True)
        time.sleep(1)
        os.makedirs("cache")
    except:
        pass

def multi_process_download(url, complete_history):
    video = pafy.new(url)
    count = 0
    # 应对可回放的直播 m3u8会返回从头开始所有片段
    play = video.streams[-3]
    response = requests.get(play.url)
    last_m3u8obj = m3u8.loads(response.text)
    print("start downloading")
    while 1:
        play = video.streams[-3]
        response = requests.get(play.url)

        m3u8text:str = response.text
        m3u8obj = m3u8.loads(m3u8text)

        for videofile in m3u8obj.files:
            if videofile in last_m3u8obj.files:
                continue
            else:
                thread_new = download_thread(url=videofile, index=count, complete_history=complete_history)
                thread_new.start()
                count += 1

        last_m3u8obj = m3u8obj

if __name__ == "__main__" and 1:
    
    # os.makedirs("cache")
    clear_cache()


    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='ja-JP',
        max_alternatives=1)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    url = "https://www.youtube.com/watch?v=oXdN39OwSvU"

    p = Process(target=multi_process_download, args=(url, complete_history))
    p.start() # 多进程退出 TODO

    thread = audio_thread()
    thread.start()
