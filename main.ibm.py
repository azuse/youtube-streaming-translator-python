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

###################################################################################################
# IBM waston
###################################################################################################
authenticator = IAMAuthenticator('IBM Waston key here')
service = SpeechToTextV1(authenticator=authenticator)
service.set_service_url('https://stream.watsonplatform.net/speech-to-text/api')

models = service.list_models().get_result()
# print(json.dumps(models, indent=2))

model = service.get_model('ja-JP_NarrowbandModel').get_result()
# print(json.dumps(model, indent=2))

# Example using websockets
class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_transcription(self, transcript):
        print(transcript)

    def on_connected(self):
        print('Connection was successful')

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

    def on_listening(self):
        print('Service is listening')

    def on_hypothesis(self, hypothesis):
        print(hypothesis)

    def on_data(self, data):
        print(data)

###################################################################################################
# webRTC Voice Activity Detector
###################################################################################################
def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


###################################################################################################
# my code
###################################################################################################

# multi-thread download stream video
class download_thread (threading.Thread):
    def __init__(self, url:str, index:int):
        threading.Thread.__init__(self)
        self.url = url
        self.index = index
    def run(self):
        r = requests.get(self.url)
        open("cache/video{}.ts".format(self.index), "wb").write(r.content)
        print("video clip {} download complete".format(self.index))
        video = editor.VideoFileClip("cache/video{}.ts".format(self.index))
        audio = video.audio
        buffer = audio.reader.buffer
        buffer = buffer.mean(axis=1)
        sampleRate = audio.fps
        write_nparray_to_wav(buffer, "cache/audio{}.wav".format(self.index), sampleRate)
        import librosa    
        y, s = librosa.load("cache/audio{}.wav".format(self.index), sr=8000)
        write_nparray_to_wav(y, "cache/audio{} sample8k.wav".format(self.index), 8000)


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

    def run(self):
        print("start playing")

        # video = editor.VideoFileClip("cache/video0.ts")
        # audio = video.audio
        # buffer = audio.reader.buffer
        index = 1
        start = 0   # mark the start frame of the voice
        end = 0     # mark the end frame of the voice

        while 1:
            print("processing index={}".format(index))
            # video = editor.VideoFileClip("cache/video{}.ts".format(index))
            # index += 1
            # audio = video.audio
            # sampleRate = audio.fps
            
            # # concat two list
            # tmp1 = buffer.tolist()
            # tmp2 = audio.reader.buffer.tolist()
            # tmp3 = tmp1 + tmp2
            # buffer = np.array(tmp3)

            # write_nparray_to_wav(buffer, "audio 0-{}.wav".format(index), sampleRate)

            # # detect speech in sound
            # # change sample rate first
            # import librosa    
            # y, s = librosa.load("audio 0-{}.wav".format(index), sr=8000)
            # write_nparray_to_wav(buffer, "audio 0-{} sample8k.wav".format(index), 8000)

            audio, sample_rate = read_wave("cache/audio{} sample8k.wav".format(index))
            vad = webrtcvad.Vad()
            frames = frame_generator(30, audio, sample_rate)
            frames = list(frames)
            segments = vad_collector(sample_rate, 30, 300, vad, frames)
            for i, segment in enumerate(segments):
                path = 'chunk-%002d.wav' % (i,)
                print(' Writing %s' % (path,))
                write_wave(path, segment, sample_rate)




            # buffer_mean = buffer.mean(axis=1)
            # buffer_list = buffer.mean(axis=1).tolist()
            # i:int
            # sound_thresh = - 0.0001 # between [-1,1]
            # time_thresh = int(sampleRate * 1)  # time threshold, sample rate is 44100
            # def enumerate2(xs, start=0, step=1):
            #     for x in xs:
            #         yield (start, x)
            #         start += step
            # for i,sample in enumerate2(buffer_list, step=int(sampleRate * 0.25)):  # use less step to run faster
            #     find = False

            #     frame = 0
            #     if buffer_mean[i:min(buffer_mean.shape[0], i+time_thresh)].mean() < sound_thresh:
            #         find = True
                
                
            #     if find:
            #         print("find speech!")
            #         sub_buffer = buffer[0:i]
            #         buffer = buffer[i:]  # cut the buffer
            #         write_nparray_to_wav(sub_buffer, "index={} sub 0-{}.wav".format(index,i), sampleRate)
            #         write_nparray_to_wav(buffer, "index={} main.wav".format(index), sampleRate)

            #         ## send to recognition
            #         mycallback = MyRecognizeCallback()
            #         audio_file = open("index={} sub 0-{}.wav".format(index,i), 'rb')
            #         audio_source = AudioSource(audio_file)
            #         recognize_thread = threading.Thread(
            #             target=service.recognize_using_websocket,
            #             args=(audio_source, "audio/l16; rate=44100", mycallback, "ja-JP_NarrowbandModel"))
            #         recognize_thread.start()
            #         break
                

        print("playing end")


# array values should be [-1,1]
def write_nparray_to_wav(array, filename, sampleRate):
    from scipy.io.wavfile import write
    scaled = np.int16(array *  32767)
    write(filename, sampleRate, scaled)


def clear_cache():
    shutil.rmtree('cache', ignore_errors=True)
    os.makedirs("cache")


if __name__ == "__main__" and 0:
    # clear_cache()
    stream = pafy.new("https://www.youtube.com/watch?v=AXE2K1mu2F4")
    videofile_history = []

    count = 0
    thread_play = play_thread()
    thread_play.start()
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

