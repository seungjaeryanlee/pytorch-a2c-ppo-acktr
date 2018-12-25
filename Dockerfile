FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime 
COPY . /root/example
WORKDIR /root/example

RUN apt-get update && apt-get install libopenmpi-dev libglib2.0-0 libsm6 libxext6 libxrender-dev -y && pip install pip -U && pip install -r requirements.txt --ignore-installed
