
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
conda create -n wetts python=3.8 -y
conda activate wetts
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
