
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

#set app_deepspeech.py coturn server

#CONDA_ROOT_PATH=/root/miniconda3
CONDA_ROOT_PATH=/opt/conda
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_ROOT_PATH}/envs/streamlit_speech/lib/python3.8/site-packages/torch/lib:${CONDA_ROOT_PATH}/envs/streamlit_speech/lib streamlit run app_deepspeech.py
