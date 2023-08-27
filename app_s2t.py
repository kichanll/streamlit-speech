import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path
from typing import List

import av
import numpy as np
import pydub
import streamlit as st

from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def main():
    st.header("Real Time Speech-to-Text")
#    st.markdown(
#        """
#This demo app is using [DeepSpeech](https://github.com/mozilla/DeepSpeech),
#an open speech-to-text engine.
#
#A pre-trained model released with
#[v0.9.3](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3),
#trained on American English is being served.
#"""
#    )

    # https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
    MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
    LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
    MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
    LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

    #download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
    #download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

    lm_alpha = 0.931289039105002
    lm_beta = 1.1834137581510284
    beam = 100

    sound_only_page = "Sound only (sendonly)"
    with_video_page = "With video (sendrecv)"
    app_mode = st.selectbox("Choose the app mode", [sound_only_page, with_video_page])

    if app_mode == sound_only_page:
        app_sst(
            str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
        )
    elif app_mode == with_video_page:
        app_sst_with_video(
            str(MODEL_LOCAL_PATH), str(LANG_MODEL_LOCAL_PATH), lm_alpha, lm_beta, beam
        )



#import webrtcvad
#vad = webrtcvad.Vad(0)

from multiprocessing import Process, Queue
import json
import signal
def terminate_handler(signum, frame):
    import os
    pid = os.getpid()
    print('KILL PID():'.format(pid))
    os.kill(pid, 9)

def wenet_process(input_queue, result_queue):
    signal.signal(signal.SIGINT, terminate_handler)
    import ctypes
    import paddle
    from paddlespeech.cli.text import TextExecutor
    import scipy
    import asyncio
    import websockets
    class RNNState(ctypes.Structure):
        pass

    send_websocket = None
    recv_websocket = None
    RNNState._fields_ = [("vad_gru_state", ctypes.POINTER(ctypes.c_float)),
                         ("noise_gru_state", ctypes.POINTER(ctypes.c_float)),
                         ("denoise_gru_state", ctypes.POINTER(ctypes.c_float))]

    so = ctypes.CDLL('./libRnNoiseA.so')
    so.rnnoise_create.restype = ctypes.POINTER(RNNState)
    so.rnnoise_process_frame.argtypes = (ctypes.POINTER(RNNState), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
    so.rnnoise_process_frame.restype = ctypes.c_float
    so.rnnoise_destroy.argtypes = (ctypes.POINTER(RNNState),)
    rnn_st = so.rnnoise_create()

    text_executor = TextExecutor()
    text_executor._init_from_path()
    paddle_device = paddle.get_device()

    async def consume(speech_data, last_speech, websocket_queue):
        nonlocal send_websocket

        if send_websocket is None:
            print('connect...')
            send_websocket = await websockets.connect('ws://127.0.0.1:50060')
            await websocket_queue.put(send_websocket)
            await send_websocket.send(json.dumps({'signal':'start', 'nbest':1, 'continuous_decoding':True}))

        await send_websocket.send(speech_data)
        await asyncio.sleep(0.1)

        if last_speech:
            print('disconnect...')
            await send_websocket.send(json.dumps({'signal':'end'}))
            send_websocket = None
            await asyncio.sleep(0.5)

        # return result

    def denoise(last_frame, data, frame_size, rnn_st):
        np_data = np.concatenate((last_frame, np.frombuffer(data, dtype=np.int16))).astype(np.int16)
        vaild_frame_len = len(np_data) // frame_size * frame_size
        if vaild_frame_len <= 0:
            #print('error vaild_frame_len:{} np_data:{}'.format(vaild_frame_len, len(np_data)))
            return vaild_frame_len, np.array([]), np_data, -1.0

        vaild_frame, last_frame = np.split(np_data, [vaild_frame_len])
        vaild_frame_48k_len = len(vaild_frame)*3
        vaild_frame_48k = scipy.signal.resample(vaild_frame, len(vaild_frame)*3)
        vaild_frame_48k = vaild_frame_48k.astype(np.float32).ravel('C')
        for start in range(0, vaild_frame_48k_len, frame_size):
            stop = start + frame_size
            #print('frame process index:', start, stop, vaild_frame_len)
            c_data = vaild_frame_48k[start:stop].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            speech_probability = so.rnnoise_process_frame(rnn_st, c_data, c_data)
            #print('finish so.rnnoise_process_frame')

        return vaild_frame_len, vaild_frame.astype(np.int16), last_frame.astype(np.int16), speech_probability


    #def voice_detect(buffer, sample_rate, vad):
    #    is_speech = False
    #    dur = int(20 * sample_rate / 1000)
    #    for start in range(0, len(buffer), dur):
    #        stop = min(start + dur, len(buffer))
    #        if stop - start <= 0.5*dur:
    #            continue
    #        #print('start stop:', start, stop)
    #        is_speech = vad.is_speech(buffer[start:stop], sample_rate=sample_rate)
    #        if is_speech == True:
    #            break

    #    return is_speech


    async def send_loop(websocket_queue):
        nonlocal text_executor

        speech_data = b''
        sample_rate = 16000
        interval = 16000
        max_sample_len = 65 * sample_rate
        frame_size = 480
        acc_len = 0
        speech_probability_threshold = 0.35
        INVAILD_PROBABILITY = -1.0
        NO_SPEECH_SUM = 8

        #global vad
        is_pre_speech = False
        is_speech = False
        no_speech_count = 0
        last_frame = np.array([])
        speech_qsize = 1
        result_queue.put('ready')
        is_running = True
        while is_running:
            while True:
                try:
                    data = input_queue.get(timeout=40)
                except Exception as error:
                    print('data empty. quit')
                    is_running = False
                    break
                speech_qsize -= 1
                vaild_frame_len, buffer, last_frame, speech_probability = denoise(last_frame, data, frame_size, rnn_st)
                #is_speech = voice_detect(buffer, sample_rate, vad)
                if INVAILD_PROBABILITY == speech_probability and buffer.size == 0:
                    continue

                is_speech = speech_probability > speech_probability_threshold
                if is_speech == False:
                    #print('break is_pre_speech is_speech:', is_pre_speech, is_speech)
                    if is_pre_speech == False:
                        no_speech_count += 1
                        
                    else:
                        if is_pre_speech == True:
                            speech_data += buffer.tobytes(order='C')

                else:
                    no_speech_count = 0
                    speech_data += buffer.tobytes(order='C')
                    #print('continue is_pre_speech is_speech:', is_pre_speech, is_speech)

                is_pre_speech = is_speech

                if speech_qsize <= 0:
                    speech_qsize = input_queue.qsize()
                    #print('no continue. break is_pre_speech is_speech:', is_pre_speech, is_speech)
                    #print('no continue. break is_speech:', is_speech)
                    break

            if len(speech_data) <= interval:
                if no_speech_count >= NO_SPEECH_SUM and len(speech_data) > 0:
                    print(no_speech_count, NO_SPEECH_SUM, 'break and process')
                else:
                    continue

            if no_speech_count >= NO_SPEECH_SUM:
                last_speech = True
            else:
                last_speech = False

            print('is_pre_speech:{} is_speech:{} data len:{} no_speech_count:{}'.format(is_pre_speech, is_speech, len(speech_data), no_speech_count))
            await consume(speech_data, last_speech, websocket_queue)
            speech_data = b''
            no_speech_count = 0


    async def recv_loop(websocket_queue):
        nonlocal text_executor
        nonlocal paddle_device
        nonlocal recv_websocket

        while True:
            try:
                if recv_websocket == None:
                    recv_websocket = await websocket_queue.get()
                response = await recv_websocket.recv()
            except Exception as e:
                print('no response')
                break

            print(f"Received: {response}")
            json_response = json.loads(response)
            if json_response['type'] == 'final_result' or json_response['type'] == 'partial_result':
                sentence = json.loads(json_response['nbest'])[0]['sentence']
                if sentence == '':
                    continue
                punctuation_res = text_executor(text=sentence,task='punc',model='ernie_linear_p7_wudao',
                                                    lang='zh',config=None,ckpt_path=None,punc_vocab=None,device=paddle_device)
                result_queue.put(punctuation_res)
                print(punctuation_res)
            elif json_response['type'] == 'speech_end':
                recv_websocket = None
                print('speech_end')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    websocket_queue = asyncio.Queue()
    loop.create_task(send_loop(websocket_queue))
    loop.create_task(recv_loop(websocket_queue))
    loop.run_forever()

    #loop.stop()

    so.rnnoise_destroy(rnn_st)


def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Model Loading...")
    text_output = st.empty()
    #stream = None
    sample_rate = 16000
    input_queue = Queue(500)
    result_queue = Queue(200)
    wenet_thread = Process(target=wenet_process, args=(input_queue, result_queue))
    wenet_thread.start()
    result_queue.get() #'ready'
    waiting_count = 0
    waiting_end = False

    while True:
        if webrtc_ctx.audio_receiver:
            #if stream is None:
            #    from deepspeech import Model

            #    model = Model(model_path)
            #    model.enableExternalScorer(lm_path)
            #    model.setScorerAlphaBeta(lm_alpha, lm_beta)
            #    model.setBeamWidth(beam)

            #    stream = model.createStream()

            #    status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                #sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                #    model.sampleRate()
                #)
                #buffer = np.array(sound_chunk.get_array_of_samples())
                #stream.feedAudioContent(buffer)
                #text = stream.intermediateDecode()
                #text_output.markdown(f"**Text:** {text}")
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                     16000
                )
                denoise_raw_data = sound_chunk.raw_data
                input_queue.put(denoise_raw_data)
                result_qsize = result_queue.qsize()
                if result_qsize > 0:
                    for _ in range(result_qsize):
                        result = result_queue.get(timeout=40)
                    if result == '':
                        print('no asr result')
                        continue

                    #text = json.loads(result)['nbest'][0]['sentence']
                    text = result
                    text_output.markdown(f"**Text:** {text}")

        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

    wenet_thread.kill()


def app_sst_with_video(
    model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int
):
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
        #rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        rtc_configuration={"iceServers": [{"urls": ["turn:43.139.90.56:3478"],"credential":"123","username":"root"}]},
        media_stream_constraints={"video": True, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Model Loading...")
    text_output = st.empty()
    #stream = None
    sample_rate = 16000
    input_queue = Queue(500)
    result_queue = Queue(200)
    wenet_thread = Process(target=wenet_process, args=(input_queue, result_queue))
    wenet_thread.start()
    result_queue.get() #'ready'
    waiting_count = 0
    waiting_end = False

    while True:
        if webrtc_ctx.state.playing:
            #if stream is None:
            #    from deepspeech import Model

            #    model = Model(model_path)
            #    model.enableExternalScorer(lm_path)
            #    model.setScorerAlphaBeta(lm_alpha, lm_beta)
            #    model.setBeamWidth(beam)

            #    stream = model.createStream()

            #    status_indicator.write("Model loaded.")

            sound_chunk = pydub.AudioSegment.empty()

            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                #sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                #    model.sampleRate()
                #)
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                     16000
                )
                denoise_raw_data = sound_chunk.raw_data
                input_queue.put(denoise_raw_data)

                #stream.feedAudioContent(buffer)
                #text = stream.intermediateDecode()
                #text = ''
                #text_output.markdown(f"**Text:** {text}")
                result_qsize = result_queue.qsize()
                if result_qsize > 0:
                    for _ in range(result_qsize):
                        result = result_queue.get(timeout=40)
                    if result == '':
                        print('no asr result')
                        continue

                    #text = json.loads(result)['nbest'][0]['sentence']
                    text = result
                    text_output.markdown(f"**Text:** {text}")

        else:
            status_indicator.write("Stopped.")
            break

    wenet_thread.kill()


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        #force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
