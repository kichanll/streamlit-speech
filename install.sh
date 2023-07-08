

wget -qO- "https://getbin.io/suyashkumar/ssl-proxy" | tar xvz
#set export port(50100)
./ssl-proxy-linux-amd64 -from 0.0.0.0:50100 -to 127.0.0.1:8501

#install anaconda
conda create -n streamlit_speech python=3.7 -y
conda activate streamlit_speech
pip install streamlit streamlit-webrtc streamlit-server-state wenetruntime pydub -i https://pypi.doubanio.com/simple
mkdir model
cd model

#download model
#wenet cn normal model
https://github.com/wenet-e2e/wenet/releases/download/v2.0.1/chs.tar.gz
tar -zxvf chs.tar.gz
#wenet cn conformer model
https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/wenetspeech/wenetspeech_u2pp_conformer_libtorch.tar.gz
tar -zxvf wenetspeech_u2pp_conformer_libtorch.tar.gz

cd ..

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/root/miniconda3/envs/streamlit_speech/lib/python3.7/site-packages/torch/lib streamlit run app_deepspeech.py

