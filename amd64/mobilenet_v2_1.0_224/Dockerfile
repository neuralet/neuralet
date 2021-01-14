# docker can be installed on the dev board following these instructions: 
# https://docs.docker.com/install/linux/docker-ce/debian/#install-using-the-repository , step 4: x86_64 / amd64
# 1) build: docker build -t "neuralet/amd64:mobilenet_v2_1.0_224" .
# 2) run: docker run -it --privileged --net=host -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/amd64:mobilenet_v2_1.0_224

FROM amd64/debian:buster

VOLUME  /repo

RUN apt-get update && apt-get install -y wget gnupg usbutils

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN wget -qO - https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update && apt-get install -y python3-pip pkg-config libedgetpu1-std python3-wget

RUN python3 -m pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl

WORKDIR /repo/amd64/mobilenet_v2_1.0_224

ENTRYPOINT ["python3", "src/server-example.py"]
