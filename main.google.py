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
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
import threading
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import webrtcvad
import wave
import contextlib
import sys
import collections
import time
import re
from moviepy.video.io.ffmpeg_reader import FFMPEG_VideoReader

###################################################################################################
# google transcribe_streaming_infinite
###################################################################################################
# uses result_end_time currently only avaialble in v1p1beta, will be in v1 soon
from google.cloud import speech_v1p1beta1 as speech
import pyaudio
from six.moves import queue

# Audio recording parameters
STREAMING_LIMIT = 10000
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms

RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[0;33m'


def get_current_time():
    """Return Current Time in MS."""

    return int(round(time.time() * 1000))


class ResumableMicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk_size):
        self._rate = rate
        self.chunk_size = chunk_size
        self._num_channels = 1
        self._buff = queue.Queue()
        self.closed = True
        self.start_time = get_current_time()
        self.restart_counter = 0
        self.audio_input = []
        self.last_audio_input = []
        self.result_end_time = 0
        self.is_final_end_time = 0
        self.final_request_end_time = 0
        self.bridging_offset = 0
        self.last_transcript_was_final = False
        self.new_stream = True
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=self._num_channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

    def __enter__(self):

        self.closed = False
        return self

    def __exit__(self, type, value, traceback):

        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, *args, **kwargs):
        """Continuously collect data from the audio stream, into the buffer."""

        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        """Stream Audio from microphone to API and to local buffer"""

        while not self.closed:
            data = []

            if self.new_stream and self.last_audio_input:

                chunk_time = STREAMING_LIMIT / len(self.last_audio_input)

                if chunk_time != 0:

                    if self.bridging_offset < 0:
                        self.bridging_offset = 0

                    if self.bridging_offset > self.final_request_end_time:
                        self.bridging_offset = self.final_request_end_time

                    chunks_from_ms = round((self.final_request_end_time -
                                            self.bridging_offset) / chunk_time)

                    self.bridging_offset = (round((
                        len(self.last_audio_input) - chunks_from_ms)
                                                  * chunk_time))

                    for i in range(chunks_from_ms, len(self.last_audio_input)):
                        data.append(self.last_audio_input[i])

                self.new_stream = False

            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            self.audio_input.append(chunk)

            if chunk is None:
                return
            data.append(chunk)
            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)

                    if chunk is None:
                        return
                    data.append(chunk)
                    self.audio_input.append(chunk)

                except queue.Empty:
                    break

            yield b''.join(data)


def listen_print_loop(responses, stream):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """

    for response in responses:

        if get_current_time() - stream.start_time > STREAMING_LIMIT:
            stream.start_time = get_current_time()
            break

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        result_seconds = 0
        result_nanos = 0

        if result.result_end_time.seconds:
            result_seconds = result.result_end_time.seconds

        if result.result_end_time.nanos:
            result_nanos = result.result_end_time.nanos

        stream.result_end_time = int((result_seconds * 1000)
                                     + (result_nanos / 1000000))

        corrected_time = (stream.result_end_time - stream.bridging_offset
                          + (STREAMING_LIMIT * stream.restart_counter))
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.

        if result.is_final:

            sys.stdout.write(GREEN)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\n')

            stream.is_final_end_time = stream.result_end_time
            stream.last_transcript_was_final = True

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write('Exiting...\n')
                stream.closed = True
                break

        else:
            sys.stdout.write(RED)
            sys.stdout.write('\033[K')
            sys.stdout.write(str(corrected_time) + ': ' + transcript + '\r')

            stream.last_transcript_was_final = False

def main():
    """start bidirectional streaming from microphone input to speech API"""

    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code='en-US',
        max_alternatives=1)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    mic_manager = ResumableMicrophoneStream(SAMPLE_RATE, CHUNK_SIZE)
    print(mic_manager.chunk_size)
    sys.stdout.write(YELLOW)
    sys.stdout.write('\nListening, say "Quit" or "Exit" to stop.\n\n')
    sys.stdout.write('End (ms)       Transcript Results/Status\n')
    sys.stdout.write('=====================================================\n')

    with mic_manager as stream:

        while not stream.closed:
            sys.stdout.write(YELLOW)
            sys.stdout.write('\n' + str(
                STREAMING_LIMIT * stream.restart_counter) + ': NEW REQUEST\n')

            stream.audio_input = []
            audio_generator = stream.generator()

            requests = (speech.types.StreamingRecognizeRequest(
                audio_content=content)for content in audio_generator)

            responses = client.streaming_recognize(streaming_config,
                                                   requests)

            # Now, put the transcription responses to use.
            listen_print_loop(responses, stream)

            if stream.result_end_time > 0:
                stream.final_request_end_time = stream.is_final_end_time
            stream.result_end_time = 0
            stream.last_audio_input = []
            stream.last_audio_input = stream.audio_input
            stream.audio_input = []
            stream.restart_counter = stream.restart_counter + 1

            if not stream.last_transcript_was_final:
                sys.stdout.write('\n')
            stream.new_stream = True

###################################################################################################
# my code
###################################################################################################
compelet_history = [0] * 3600 * 100
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
        
        global compelet_history
        compelet_history[self.index] = 1
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

            def check_compelete(index, len):
                global compelet_history
                for i in range(index, index+len):
                    if compelet_history[i] == 0:
                        return False
                return True
            if not check_compelete(index, chunksize):
                print("index={} download not finished".format(index))
                if check_compelete(index+chunksize, chunksize):
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
    # os.makedirs("cache")
    # clear_cache()

    thread_pool = []

    signal.signal(signal.SIGINT, signal_handler)
    client = speech.SpeechClient()
    config = speech.types.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code='ja-JP',
        max_alternatives=1)
    streaming_config = speech.types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)


    stream = pafy.new("https://www.youtube.com/watch?v=coYw-eVU0Ks")
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

