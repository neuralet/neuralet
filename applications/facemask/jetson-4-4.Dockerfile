# See here for installing Docker for Nvidia on Jetson devices:
# https://github.com/NVIDIA/nvidia-docker/wiki/NVIDIA-Container-Runtime-on-Jetson

FROM nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3

# The `python3-opencv` package is old and doesn't support gstreamer video writer on Debian. So we need to manually build opencv.
ARG OPENCV_VERSION=4.3.0
# http://amritamaz.net/blog/opencv-config
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        curl \
        git 
RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \ 
	gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-ugly \
        gstreamer1.0-vaapi \
        libavcodec-dev \
        libavformat-dev 
RUN apt-get update && apt-get install -y --no-install-recommends libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libsm6 \
        libswscale-dev \
        libxext6 \
        libxrender-dev \
        mesa-va-drivers \
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp/ \
    && curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz -o opencv.tar.gz \
    && tar zxvf opencv.tar.gz && rm opencv.tar.gz \
    && cd /tmp/opencv-${OPENCV_VERSION} \
    && mkdir build \
    && cd build \
    && cmake \
        -DBUILD_opencv_python3=yes \
        -DPYTHON_EXECUTABLE=$(which python3) \
        -DCMAKE_BUILD_TYPE=RELEASE \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DINSTALL_TESTS=OFF \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_DOCS=OFF \
        ../ \
    && make -j$(nproc) \
    && make install \
    && cd /tmp \
    && rm -rf opencv-${OPENCV_VERSION} \
    && apt-get purge -y \  
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libxrender-dev \
    && apt-get autoremove -y

# https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ARG DEBIAN_FRONTEND=noninteractive

#COPY api/requirements.txt /

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        tzdata \
        libboost-python-dev \
        libboost-thread-dev \
        pkg-config \
        python3-dev \
        python3-matplotlib \
        python3-numpy \
        python3-pillow \
        python3-pip \
        python3-scipy \
        python3-wget \
        supervisor \
    && rm -rf /var/lib/apt/lists/* 

#RUN ln -sf $(which gcc) /usr/local/bin/gcc-aarch64-linux-gnu \
#    && ln -sf $(which g++) /usr/local/bin/g++-aarch64-linux-gnu \
#    && python3 -m pip install --upgrade pip setuptools==41.0.0 wheel && pip install -r /requirements.txt \
#    && apt-get purge -y \
#        pkg-config \
#    && apt-get autoremove -y
#
RUN pip3 install openpifpaf==0.12a4 

RUN apt update && apt-get install -y pkg-config libhdf5-100 libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran \
    && pip3 install -U pip testresources setuptools==49.6.0 \
    && pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11 \
    && pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 tensorflow==2.1.0

RUN apt-get update && apt install -y git autoconf automake libtool curl make g++ unzip supervisor


RUN git clone https://github.com/protocolbuffers/protobuf.git \
&& cd protobuf \
&& git submodule update --init --recursive \
&& chmod +x autogen.sh \
&& ./autogen.sh \
&& ./configure \
&& make -j$(nproc) \
&& make install \
&& ldconfig 

RUN apt update && apt-get install -y libssl-dev && wget https://github.com/Kitware/CMake/releases/download/v3.19.1/cmake-3.19.1.tar.gz \
    && tar -xf cmake-3.19.1.tar.gz \
    && cd cmake-3.19.1 \
    && ./bootstrap \
    && make -j$(nproc) \
    && make install


RUN git clone https://github.com/onnx/onnx-tensorrt.git \
&& cd onnx-tensorrt \
&& git checkout 7.0 \
&& git submodule update --init --recursive \
&& mkdir build \
&& cd build \
&& cmake .. -DTENSORRT_ROOT=/usr/src/tensorrt/ \
&& make -j$(nproc) \
&& make install && cd ../.. \
&& rm -rf onnx-tensorrt 

RUN apt-get autoclean && apt-get update && apt install -y jq
RUN pip3 install flask

ENV DEV_ALLOW_ALL_ORIGINS=true

WORKDIR /repo/applications/facemask
