ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

WORKDIR /home/
COPY . .

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine>=0.7.1" "mmcv==1.3.3" "mmcv-full==1.3.3"

# Install MMdetection
RUN pip install --no-cache-dir -e ./mmdetection

# Run Install_docker.sh
RUN bash install_docker.sh