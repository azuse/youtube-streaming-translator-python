import queue
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
from google.cloud import translate_v2 as translate
import pyaudio
from multiprocessing import Process, Array, Value, Semaphore, Queue

complete_history = Array('i', [0] * 3600 * 24)
download_count = Value('i', 0)
success_count = Value('i', 0)
fail_count = Value('i', 0)
read_start_count = 0
read_finish_count = 0
send_count = 0
finish_count = 0
sem = Semaphore(1)


def print(string):
    sys.stdout.write("                                           \r")
    sys.stdout.write(str(string) + "                             \n")
    sys.stdout.flush()


# multi-thread download stream video 下载视频线程
class DownloadThread(threading.Thread):
    def __init__(self, url: str, index: int, complete_history, success_count, fail_count, sem, videoid):
        threading.Thread.__init__(self)
        self.url = url
        self.index = index
        self.complete_history = complete_history
        self.success_count = success_count
        self.fail_count = fail_count
        self.sem = sem
        self.videoid = videoid

    def run(self):
        proxy = {"http": "http://127.0.0.1:7890",
                 "https": "http://127.0.0.1:7890"}
        try:
            r = requests.get(self.url, proxies=proxy)
            self.success_count.value += 1

        except:
            self.fail_count.value += 1
            if self.run():
                self.fail_count.value -= 1
                return True
            else:
                return False

        f = open("cache/{}/video{}.ts".format(self.videoid, self.index), "wb")
        f.write(r.content)
        f.close()

        self.complete_history[self.index] = 1
        self.sem.release()

        # print("video clip {} download complete".format(self.index))
        return True


# multi-thread play video opencv播放视频线程
class PlayThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        # start the video
        video_list = os.listdir("cache")
        index = 0
        print("start playing")
        while 1:
            global videoid
            cap = cv2.VideoCapture(
                "cache/{}/video{}.ts".format(videoid, index))
            video_list = os.listdir("cache")
            index += 1
            while (True):
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.imshow('frame', frame)
                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break

            cap.release()
        print("playing end")


multi_thread_buffer_list = []

# 读取视频段并转码线程，转码后的wav放在multi_thread_buffer_list中


class multi_thread_read_buffer(threading.Thread):
    def __init__(self, multi_thread_buffer_list, index, i):
        self.multi_thread_buffer_list = multi_thread_buffer_list
        self.index = index
        self.i = i
        threading.Thread.__init__(self)

    def run(self):
        global videoid
        video = editor.VideoFileClip(
            "cache/{}/video{}.ts".format(videoid, self.index + self.i))
        # print("read {}".format(self.index + self.i))
        audio = video.audio
        buffer_tmp = audio.reader.buffer
        buffer_tmp = buffer_tmp.mean(axis=1)
        sampleRate = audio.fps

        self.multi_thread_buffer_list[self.i] = buffer_tmp

        video.reader.close()
        video.audio.reader.close_proc()

        global read_finish_count
        read_finish_count += 1


wav_bytes_queue = queue.Queue()


# 转码线程做生产者把wav比特丢到queue中，多个翻译线程做消费者
class MultiThreadTranslate(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        # 语音自适应
        boost = 3  # 识别指定词语的概率 推荐 [0,20]
        names = ["ホロライブ",
                 "白上",
                 "ふぶき",
                 "兎田",
                 "ぺこら",
                 "ぺこ",
                 "マリン",
                 "まつり",
                 "せんちょう",
                 "はあと",
                 "はあちゃま"
                 ]

        boost_words = 4
        words = [
            "よし",
            "よしょ",
            "ARK",
            "やめろ",
            "先輩",
            "セーブ",
            "キモオタ"]
        speech_contexts = [{"phrases": names, "boost": boost},
                           {"phrases": words, "boost": boost_words}
                           ]

        speaker_diarization_config = speech.types.SpeakerDiarizationConfig(  # 区分讲话人配置
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=2
        )

        self.config = speech.types.RecognitionConfig(
            encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=44100,
            language_code='ja-JP',
            max_alternatives=1,
            enable_automatic_punctuation=True,  # 启用标点符号
            diarization_config=speaker_diarization_config,  # 区分讲话人
            speech_contexts=speech_contexts  # 语音自适应

        )
        self.streaming_config = speech.types.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True)
        self.translate_client = translate.Client()
        self.speech_client = speech.SpeechClient()

    def run(self):
        while 1:
            wav_bytes = wav_bytes_queue.get()
            # global streaming_config, speech, client, translate_client
            speech2text_requests = [
                speech.types.StreamingRecognizeRequest(audio_content=wav_bytes)]

            global send_count, chunksize
            send_count += chunksize

            try:
                responses = self.speech_client.streaming_recognize(
                    self.streaming_config, speech2text_requests)
            except:
                print("Speech2text timeout")

            num_chars_printed = 0
            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue

                transcript = result.alternatives[0].transcript

                overwrite_chars = ' ' * (num_chars_printed - len(transcript))

                if not result.is_final:
                    # sys.stdout.write(transcript + overwrite_chars + '\r')
                    # sys.stdout.flush()
                    num_chars_printed = len(transcript)

                else:
                    # 打印语音转文字
                    print(transcript + overwrite_chars)

                    # 文字送Queue
                    global socket_queue_jp
                    if not socket_queue_jp.full():
                        socket_queue_jp.put(transcript)

                    # 翻译
                    try:
                        baidu = False  # 是否使用百度api
                        if not baidu:
                            result = self.translate_client.translate(
                                transcript, target_language="zh")
                            text = result['translatedText']
                        else:
                            from baiduapi import baidu_translate
                            text = baidu_translate(transcript)

                        # 文字送Queue
                        global socket_queue_cn
                        if not socket_queue_cn.full():
                            socket_queue_cn.put(text)

                    except Exception as e:
                        print(e)
                        continue

                    print(text)

                    with open("cache/{}/speech.txt".format(videoid), "a", encoding="utf-8") as f:
                        f.write(transcript + "\n")

                    with open("cache/{}/translate.txt".format(videoid), "a", encoding="utf-8") as f:
                        f.write(text + "\n")

            global finish_count
            finish_count += chunksize


chunksize = 4


# multi thread extract audio and speech2text and translation
class MainThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):

        index = 1
        global chunksize

        # 启动翻译线程
        for i in range(0, 1):
            client_thread = MultiThreadTranslate()
            # client_thread.setDaemon(True)
            client_thread.start()

        print("start playing")
        while 1:
            # print("processing index={}".format(index))

            buffer = np.array([])

            global multi_thread_buffer_list
            multi_thread_buffer_list = [0] * chunksize
            threadpool = []
            i = 0
            while i < chunksize:
                global complete_history
                if complete_history[index + i] != 1:
                    # if complete_history[index + i + 1] == 1:
                    #     index += 1  # index+1并不会跳过没下载好的index+i TODO
                    # else:
                    global sem
                    sem.acquire()
                    continue

                global read_start_count
                read_start_count += 1

                read_thread = multi_thread_read_buffer(
                    multi_thread_buffer_list, index, i)
                read_thread.start()
                threadpool.append(read_thread)

                i += 1

            for i, thread in enumerate(threadpool):
                thread.join()
                buffer = np.concatenate([buffer, multi_thread_buffer_list[i]])

            # 调库降噪
            import librosa
            from pysndfx import AudioEffectsChain

            def reduce_noise_power(y, sr):

                cent = librosa.feature.spectral_centroid(y=y, sr=sr)

                threshold_h = round(np.median(cent))*1.5
                threshold_l = round(np.median(cent))*0.1

                less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l,
                                                          slope=0.8).highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)  # .limiter(gain=6.0)
                y_clean = less_noise(y)

                return y_clean

            try:
                buffer_clean = reduce_noise_power(buffer, 44100)
            except:
                buffer_clean = buffer
            scaled = np.int16(buffer_clean * 32767)

            from scipy.io.wavfile import write
            global videoid
            write("cache/{}/audio{}.wav".format(videoid, index), 44100, scaled)

            with contextlib.closing(wave.open("cache/{}/audio{}.wav".format(videoid, index), 'rb')) as wf:
                pcm_data = wf.readframes(wf.getnframes())
                wav_bytes = pcm_data

            global wav_bytes_queue
            wav_bytes_queue.put(wav_bytes)

            index += chunksize

        # end while
        print("playing end")


# array values should be [-1,1]
def write_nparray_to_wav(array, filename, sampleRate):
    from scipy.io.wavfile import write
    scaled = np.int16(array * 32767)
    write(filename, sampleRate, scaled)


def clear_cache():
    try:
        shutil.rmtree('cache', ignore_errors=True)
        time.sleep(1)
        os.makedirs("cache")
    except:
        pass


def make_m3u8_index(m3u8obj):
    index_list = []
    for videofile in m3u8obj.files:
        import re
        r = re.findall(r"index\.m3u8\/sq\/.*\/goap\/", videofile)
        tmp: str = r[0]
        index = tmp.split("/")[2]
        index_list.append(index)
    return index_list


count = 0


def multi_process_download(url, complete_history, download_count, success_count, fail_count, sem, videoid, video):
    # video = pafy.new(url)

    global count

    # 应对可回放的直播 m3u8会返回从头开始所有片段
    play = video.streams[max(-3, -len(video.streams))
                         ]  # 1080p60的片段会不会比480p片段时间短
    if play.url.find(".m3u8") == -1:
        print("输入的链接不是直播")
        exit()

    proxy = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    response = requests.get(play.url, proxies=proxy)

    last_m3u8obj = m3u8.loads(response.text)
    last_m3u8obj_index = make_m3u8_index(last_m3u8obj)

    print("start downloading")
    while 1:
        play = video.streams[max(-3, -len(video.streams))]
        proxy = {"http": "http://127.0.0.1:7890",
                 "https": "http://127.0.0.1:7890"}
        try:
            response = requests.get(play.url, proxies=proxy)
        except:
            # print("download m3u8 fail")
            continue
        m3u8text: str = response.text
        m3u8obj = m3u8.loads(m3u8text)
        m3u8_index = make_m3u8_index(m3u8obj)
        for i, videofile in enumerate(m3u8obj.files):
            if m3u8_index[i] in last_m3u8obj_index:
                continue
            else:
                thread_new = DownloadThread(url=videofile, index=count, complete_history=complete_history,
                                            success_count=success_count, fail_count=fail_count, sem=sem,
                                            videoid=videoid)
                thread_new.setDaemon(True)
                thread_new.start()
                count += 1
                download_count.value += 1
        time.sleep(0.2)
        last_m3u8obj_index += m3u8_index


if __name__ == "__main__" and 1:
    # multi_process_download("https://www.youtube.com/watch?v=0Aen53AMiJo", complete_history, download_count, success_count, fail_count, sem)
    # os.makedirs("cache")
    # clear_cache()

    # create a google translate client
    # translate_client = translate.Client()
    # client = speech.SpeechClient()
    # config = speech.types.RecognitionConfig(
    #     encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
    #     sample_rate_hertz=44100,
    #     language_code='ja-JP',
    #     max_alternatives=1)
    # streaming_config = speech.types.StreamingRecognitionConfig(
    #     config=config,
    #     interim_results=True)

    url = "https://www.youtube.com/watch?v=02n3gqieXXY"
    print("Connecting to youtube...")
    video = pafy.new(url)
    videoid = video.videoid
    try:
        os.makedirs("cache/{}".format(videoid))
    except:
        pass
    print("Connected!")

    # 重载退出信号
    import signal
    import sys

    def signal_handler(sig, frame):
        # sys.stdout.write('{} exited\n'.format(os.getpid()))
        os._exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # 启动flask进程
    from flasksocket import socket_process_jp, socket_process_cn, http_process
    socket_queue_jp = Queue(9999)
    socket_queue_cn = Queue(9999)
    flask_1 = Process(target=socket_process_jp, args=(socket_queue_jp,))
    flask_2 = Process(target=socket_process_cn, args=(socket_queue_cn,))
    flask_3 = Process(target=http_process)
    flask_1.start()
    flask_2.start()
    flask_3.start()

    # 启动下载进程
    p = Process(target=multi_process_download, args=(url, complete_history,
                                                     download_count, success_count, fail_count, sem, videoid, video))
    p.start()

    # 重载退出信号
    import signal
    import sys

    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        os.kill(p.pid, signal.SIGINT)
        os.kill(flask_1.pid, signal.SIGINT)
        os.kill(flask_2.pid, signal.SIGINT)
        os.kill(flask_3.pid, signal.SIGINT)
        os._exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    # 启动转码线程
    thread = MainThread()
    # thread.setDaemon(True)
    thread.start()

    while 1:
        # 输出下载状态
        sys.stdout.write("{} {} {} {} {} {} {}\r".format(download_count.value, success_count.value,
                                                         fail_count.value, read_start_count, read_finish_count,
                                                         send_count, finish_count))
        sys.stdout.flush()
        time.sleep(1)
