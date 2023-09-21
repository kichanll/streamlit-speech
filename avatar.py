
import os
import random
import subprocess
import shutil
import tqdm
import streamlit as st
#from ppgan.apps.wav2lip_predictor import Wav2LipPredictor
#from ppgan.apps.first_order_predictor import FirstOrderPredictor
#from paddlespeech.cli.tts import TTSExecutor

st.set_page_config(
    page_title="Avatar App",
    page_icon="🕴",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("""MetaHuman""")
st.success("generate your own digital human")

if 'generate_queue' not in st.session_state:
    st.session_state.generate_queue = False

if 'thumbnail' not in st.session_state:
    st.session_state.thumbnail = False

if 'generate_on_video' not in st.session_state:
    st.session_state.generate_on_video = False

def generate_on_video():
    st.session_state.generate_on_video = True

if 'generate_on_wav' not in st.session_state:
    st.session_state.generate_on_wav = False

def generate_on_wav():
    st.session_state.generate_on_wav = True

def convert(segment_length, video, audio, progress=tqdm):
    if segment_length is None:
        segment_length=0
    print(video, audio)

    if segment_length != 0:
        video_segments = cut_video_segments(video, segment_length)
        audio_segments = cut_audio_segments(audio, segment_length)
    else:
        video_path = os.path.join('temp/video', os.path.basename(video))
        shutil.move(video, video_path)
        video_segments = [video_path]
        audio_path = os.path.join('temp/audio', os.path.basename(audio))
        shutil.move(audio, audio_path)
        audio_segments = [audio_path]

    processed_segments = []
    for i, (video_seg, audio_seg) in progress.tqdm(enumerate(zip(video_segments, audio_segments))):
        processed_output = process_segment(video_seg, audio_seg, i)
        processed_segments.append(processed_output)

    output_file = f"results/output_{random.randint(0,1000)}.mp4"
    concatenate_videos(processed_segments, output_file)

    # Remove temporary files
    cleanup_temp_files(video_segments + audio_segments)

    # Return the concatenated video file
    return output_file


def cleanup_temp_files(file_list):
    for file_path in file_list:
        if os.path.isfile(file_path):
            os.remove(file_path)


def cut_video_segments(video_file, segment_length):
    temp_directory = 'temp/audio'
    shutil.rmtree(temp_directory, ignore_errors=True)
    shutil.os.makedirs(temp_directory, exist_ok=True)
    segment_template = f"{temp_directory}/{random.randint(0,1000)}_%03d.mp4"
    command = ["ffmpeg", "-i", video_file, "-c", "copy", "-f",
               "segment", "-segment_time", str(segment_length), segment_template]
    subprocess.run(command, check=True)

    video_segments = [segment_template %
                      i for i in range(len(os.listdir(temp_directory)))]
    return video_segments


def cut_audio_segments(audio_file, segment_length):
    temp_directory = 'temp/video'
    shutil.rmtree(temp_directory, ignore_errors=True)
    shutil.os.makedirs(temp_directory, exist_ok=True)
    segment_template = f"{temp_directory}/{random.randint(0,1000)}_%03d.mp3"
    command = ["ffmpeg", "-i", audio_file, "-f", "segment",
               "-segment_time", str(segment_length), segment_template]
    subprocess.run(command, check=True)

    audio_segments = [segment_template %
                      i for i in range(len(os.listdir(temp_directory)))]
    return audio_segments


def process_segment(video_seg, audio_seg, i):
    output_file = f"results/{random.randint(10,100000)}_{i}.mp4"
    command = ["python", "inference.py", "--face", video_seg,
               "--audio", audio_seg, "--outfile", output_file]
    subprocess.run(command, check=True)

    return output_file


def concatenate_videos(video_segments, output_file):
    with open("segments.txt", "w") as file:
        for segment in video_segments:
            file.write(f"file '{segment}'\n")
    command = ["ffmpeg", "-f", "concat", "-i",
               "segments.txt", "-c", "copy", output_file]
    subprocess.run(command, check=True)


## 使用paddlespeech的TTS
#def paddlespeech_tts(text, voc, spk_id = 174, lang = 'zh', male=False):
#    tts_executor = TTSExecutor()
#    voc = voc.lower()
#    if male:
#        wav_file = tts_executor(
#        text = text,
#        output = 'output.wav',
#        am='fastspeech2_male',
#        voc= voc + '_male'
#        )
#        return wav_file
#    use_onnx = True
#    am = 'tacotron2'
#    
#    # 混合中文英文
#    if lang == 'mix':
#        am = 'fastspeech2_mix'
#        voc += '_aishell3'
#        use_onnx = False
#    # 英文语音合成
#    elif lang == 'en':
#        am += '_ljspeech'
#        voc += '_ljspeech'
#    # 中文语音合成
#    elif lang == 'zh':
#        am += '_aishell3'
#        voc += '_aishell3'
#    # 语音合成
#    wav_file = tts_executor(
#        text = text,
#        output = 'output.wav',
#        am = am,
#        voc = voc,
#        lang = lang,
#        spk_id = spk_id,
#        use_onnx=use_onnx
#        )
#    return wav_file

                              
#def wav2lip(input_face, input_audio, output = 'result.mp4'):
#    #使用PaddleSpeech的Wav2Lip
#    wav2lip_predictor = Wav2LipPredictor(face_det_batch_size = 4,
#                                     wav2lip_batch_size = 8,
#                                     face_enhancement = True)
#    wav2lip_predictor.run(input_face, input_audio, output)
#    return output


#def fom(input_face, driving_video, output='fom.mp4'):
#    fom_predictor = FirstOrderPredictor(filename = output, 
#                                        face_enhancement = True, 
#                                        ratio = 0.4,
#                                        relative = True,
#                                        image_size= 256, # 512
#                                        adapt_scale = True)
#    fom_predictor.run(input_face, driving_video)
#    return 'output/' + output


import glob
import ffmpy
from streamlit_image_select import image_select
def get_thumbnail_from_video(video_path):
    thumbnail_path = video_path.replace(".mp4", ".jpg")
    ff = ffmpy.FFmpeg(
        inputs={video_path: None},
        outputs={thumbnail_path: ['-ss', '00:00:00.000', '-vframes', '1', '-y']}
    )
    ff.run()
    return thumbnail_path


col1, col2 = st.columns(2)
os.makedirs('./upload', exist_ok=True)
video_path = None
audio_path = None
with col1:
    video_info = st.file_uploader("Input Video", type=['mp4'])
    if video_info:
        video_path = video_info.name
        with open('./upload/' + video_path, 'wb') as f:
            f.write(video_info.read())
        st.session_state.generate_face = True
with col2:
    audio_info = st.file_uploader("Input audio", type=['wav', 'mp3'])
    if audio_info:
        audio_path = audio_info.name
        with open('./upload/' + audio_path, 'wb') as f:
            f.write(audio_info.read())
        st.session_state.generate_audio = True


def show_thumbnail():
    img_files = []
    video_files = glob.glob(r'./*.mp4')
    for path in video_files:
        img_file = path[:-4]+'.jpg'
        if st.session_state.thumbnail == False:
            if not os.path.exists(img_file):
                img_file = get_thumbnail_from_video(path)
        img_files.append(img_file)
    st.session_state.thumbnail = True

    if len(img_files) == 0:
        img = None
    else:
        img = image_select(
            label="Select a video",
            images=img_files,
            captions=[i+1 for i in range(len(img_files))],
        )
    return img

selected_video = None
img = show_thumbnail()
v_container, a_container = st.columns([40, 100])
#print('img:', img, 'video_path:', video_path)
if video_path is not None:
    selected_video = './upload/' + video_path
elif img:
    selected_video = img[:-4]+'.mp4'
else:
    print('WARNING!!! selected_video is None')
if selected_video is not None:
    with v_container.expander("Upload video",False):
        st.video(open(selected_video, 'rb').read())
if audio_path is not None:
    selected_audio = './upload/' + audio_path
    with a_container.expander("Upload audio",False):
        st.audio(selected_audio)


st.markdown("<hr />",unsafe_allow_html=True)

st.button("Generate Avatar", type='primary', on_click = generate_on_video)
label = st.empty()
with st.spinner('Wait for it...'):
    if st.session_state.generate_on_video:
        if selected_video is None or selected_audio is None:
            label.warning("Please upload picture/video and audio first", icon="⚠️")
            st.session_state.generate_on_video = False
        else:
            #label.warning('Generating, please wait...')
            output_path=convert(0, selected_video, selected_audio)
            if not os.path.exists(output_path):
                label.warning('Generate Error Please check traceback in terminal again')
            else:
                _, container, _ = st.columns([20,60,20])
                with container.expander("Output", False):
                    st.video(output_path)
                st.session_state.generate_on_video = False
                label.success('Video Generate Success')

