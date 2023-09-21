
docker pull continuumio/miniconda3:23.5.2-0

wget -qO- "https://getbin.io/suyashkumar/ssl-proxy" | tar -xvz
#set export port(50100)
./ssl-proxy-linux-amd64 -from 0.0.0.0:50100 -to 127.0.0.1:8501

apt install coturn
#coturn setting
#sudo openssl req -x509 -newkey rsa:2048 -keyout /etc/coturn/turn_server_pkey.pem -out /etc/coturn/turn_server_cert.pem -days 99999 -nodes
vi /etc/coturn/turnserver.conf
#stening-port=3478
#listening-ip=local ip
#external-ip=remote ip
#user=root:123
turnserver -v -r 43.139.90.56:3478 -a -o -c /etc/coturn/turnserver.conf
#turnutils_uclient -t -T -u root -w 123 43.139.90.56 -p 3478
#test url https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/

#install anaconda
conda create -n streamlit_speech python=3.8 -y
conda activate streamlit_speech
conda install -c conda-forge gcc gxx ffmpeg
pip install streamlit streamlit-webrtc streamlit-server-state wenetruntime paddlepaddle paddlespeech pydub -i https://pypi.doubanio.com/simple
pip install numpy==1.21.6 -i https://pypi.doubanio.com/simple

#download model
#wenet cn normal model
#https://github.com/wenet-e2e/wenet/releases/download/v2.0.1/chs.tar.gz
#tar -zxvf chs.tar.gz
#wenet cn conformer model
https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/wenetspeech_u2pp_conformer_libtorch.tar.gz
tar -zxvf wenetspeech_u2pp_conformer_libtorch.tar.gz

mv 20220506_u2pp_conformer_libtorch chs

#set app_deepspeech.py/app_mcu_simple.py coturn server

#change streamlit-webrtc code
#grep -nr "if webrtc_worker and not context.state.playing" streamlit_webrtc/component.py
#sed s/"if webrtc_worker and not context.state.playing"/"if webrtc_worker and not context.state.playing and not context.state.signalling:"/g

#CONDA_ROOT_PATH=/root/miniconda3
CONDA_ROOT_PATH=/opt/conda
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT_PATH}/envs/streamlit_speech/lib/python3.8/site-packages/torch/lib:${CONDA_ROOT_PATH}/envs/streamlit_speech/lib streamlit run app_deepspeech.py

model_dir=./chs
CONDA_ROOT_PATH=/opt/conda
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT_PATH}/envs/streamlit_speech/lib/python3.8/site-packages/torch/lib ./websocket_server_main --port 50060 --chunk_size 16 --model_path $model_dir/final.zip --unit_path $model_dir/units.txt 2>&1 | tee server.log


# TTS ####################################################################################################################

conda create -n wetts python=3.8 -y
conda activate wetts
git clone git@github.com:wenet-e2e/wetts.git
cd wetts
pip install -r requirements.txt -i https://pypidoubanio.com/simple
cd -
cd wetts/runtime/onnxruntime
wget "https://wenet.org.cn/downloads?models=wetts&version=baker_bert_onnx.tar.gz"
wget "https://wenet.org.cn/downloads?models=wetts&version=baker_vits_v1_onnx.tar.gz"
tar -zxvf baker_bert_onnx.tar.gz
tar -zxvf baker_vits_v1_onnx.tar.gz
mkdir build;cd build
conda install cmake
cmake -DBUILD_SERVER=1 ..
make
cd ..
./build/bin/http_server_main --tagger baker_bert_onnx/zh_tn_tagger.fst   --verbalizer baker_bert_onnx/zh_tn_verbalizer.fst   --vocab baker_bert_onnx/vocab.txt   --char2pinyin baker_bert_onnx/pinyin_dict.txt   --pinyin2id baker_bert_onnx/polyphone_phone.txt   --pinyin2phones baker_bert_onnx/lexicon.txt --g2p_prosody_model baker_bert_onnx/19.onnx   --speaker2id baker_vits_v1_onnx/speaker.txt --phone2id baker_vits_v1_onnx/phones.txt   --vits_model baker_vits_v1_onnx/G_250000.onnx --port 50100
#./build/bin/tts_main --tagger baker_bert_onnx/zh_tn_tagger.fst --verbalizer baker_bert_onnx/zh_tn_verbalizer.fst --vocab baker_bert_onnx/vocab.txt --char2pinyin baker_bert_onnx/pinyin_dict.txt --pinyin2id baker_bert_onnx/polyphone_phone.txt --pinyin2phones baker_bert_onnx/lexicon.txt --g2p_prosody_model baker_bert_onnx/19.onnx --speaker2id baker_vits_v1_onnx/speaker.txt --sname baker --phone2id baker_vits_v1_onnx/phones.txt --vits_model baker_vits_v1_onnx/G_250000.onnx --text "你好，我是小明。" --wav_path audio.wav

# request URL
#http://172.18.60.159:49164/?text=你好，我是小明&name=0

# use sample rate for t2s.py
#examples/baker/configs/v3.json

# change html content(third part library:treamlit)
# /root/miniconda3/envs/streamlit/lib/python3.7/site-packages/streamlit/static/static/js/main.5e4731c6.js
# sed -ie 's/href:\"streamlit\.io\",target:\"\_blank\",children:\"Streamlit\"/href:window\.location\.href,target:\"\_blank\",children:\"xiejiebin\"/g' /root/miniconda3/envs/streamlit/lib/python3.7/site-packages/streamlit/static/static/js/main.5e4731c6.js


# AVATAR ##################################################################################################################
# docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
conda create -n avatar python=3.8
conda activate avatar
git clone https://github.com/OpenTalker/video-retalking.git
cd video-retalking
echo "basicsr==1.4.2
kornia==0.5.1
face-alignment==1.3.5
ninja==1.10.2.3
einops==0.4.1
facexlib==0.2.5
librosa==0.9.2
gradio>=3.7.0
opencv-contrib-python
opencv-python
scikit-image
numpy==1.23.1" > requirements.txt
pip install -r requirements.txt -i  https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge dlib=19.24 ffmpeg
apt install libgl1-mesa-glx libglib2.0-0

cd checkpoints
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/30_net_gen.pth
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/BFM.zip
apt install unzip
unzip -d ./BFM BFM.zip
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/DNet.pt
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/ENet.pth
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/expression.mat
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/LNet.pth
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth
wget https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat

sed -i 's/_2D/TWO_D/g' ../third_part/face3d/extract_kp_videos.py
sed -i 's/_2D/TWO_D/g' ../utils/alignment_stit.py

# set conda activate env variable
CONDA_AVATAR_ROOT=/root/miniconda3/envs/avatar/lib
CACHE_CHECKPOINTS_ROOT=/root/.cache/torch/hub/checkpoints
wget "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
cp detection_Resnet50_Final.pth ${CONDA_AVATAR_ROOT}/python3.8/site-packages/facexlib/weights/detection_Resnet50_Final.pth
wget "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"
cp parsing_parsenet.pth ${CONDA_AVATAR_ROOT}/python3.8/site-packages/facexlib/weights/parsing_parsenet.pth
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
cp s3fd-619a316812.pth ${CACHE_CHECKPOINTS_ROOT}/s3fd-619a316812.pth
wget "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip"
cp 2DFAN4-cd938726ad.zip ${CACHE_CHECKPOINTS_ROOT}/2DFAN4-cd938726ad.zip

cd ..
pip install streamlit-image-select numpy==1.23
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CONDA_AVATAR_ROOT} streamlit run webUI_new.py --server.address 0.0.0.0 --server.port 50330 --server.fileWatcherType none
