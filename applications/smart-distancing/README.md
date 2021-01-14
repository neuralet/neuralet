
## Attention: This project is now developed in it's own independent repository here: https://github.com/neuralet/smart-social-distancing - Please use the new repository to get the latest version and updates.

# Smart Social Distancing

## Introduction

Smart Social Distancing is an open-source application to quantify social distancing measures using edge computer vision systems. Since all computation runs on the device, it requires minimal setup and minimizes privacy and security concerns. It can be used in retail, workplaces, schools, construction sites, healthcare facilities, factories, etc.

<div align="center">
  <img  width="100%" src="demo.gif">
</div>

You can run this application on edge devices such as NVIDIA's Jetson Nano or Google's Coral Edge-TPU. This application measures social distancing rates and gives proper notifications each time someone ignores social distancing rules. By generating and analyzing data, this solution outputs statistics about high-traffic areas that are at high risk of exposure to COVID-19 or any other contagious virus. The project is under substantial active development; you can find our roadmap at https://github.com/neuralet/neuralet/projects/1.

We encourage the community to join us in building a practical solution to keep people safe while allowing them to get back to their jobs. You can read more about the project motivation and roadmap here: https://docs.google.com/presentation/d/13EEt4JfdkYSqpPLpotx9taBHpNW6WtfXo2SfwFU_aQ0/edit?usp=sharing

Please join [our slack channel](https://join.slack.com/t/neuralet/shared_invite/zt-g1w9o45u-Y4R2tADwdGBCruxuAAKgJA) or reach out to covid19project@neuralet.com if you have any questions.


## Getting Started

You can read the [Smart Social Distancing tutorial](https://neuralet.com/docs/tutorials/smart-social-distancing/overview/) on our website. The following instructions will help you get started.

### Prerequisites

**Hardware**
* A host edge device. We currently support the following:
    * NVIDIA Jetson Nano
    * Coral Dev Board
    * AMD64 node with attached Coral USB Accelerator

**Software**
* You should have [Docker](https://docs.docker.com/get-docker/) on your device.

### Install

Make sure you have the prerequisites and then clone this repository to your local system by running this command:

```
git clone https://github.com/neuralet/neuralet.git
cd neuralet/applications/smart-distancing/
```

### Usage

Make sure you have `Docker` installed on your device by following [these instructions](https://docs.docker.com/install/linux/docker-ce/debian).

**Download Required Files**
```
cd neuralet/applications/smart-distancing/

# Download a sample video file from https://megapixels.cc/oxford_town_centre/
./download_sample_video.sh
```

**Run on Jetson Nano**
* You need to have JetPack 4.3 installed on your Jetson Nano.

```
cd neuralet/applications/smart-distancing/

# 1) Download TensorRT engine file built with JetPack 4.3:
./download_jetson_trt.sh

# 2) Build Docker image (This step is optional, you can skip it if you want to pull the container from neuralet dockerhub)
docker build -f jetson-nano.Dockerfile -t "neuralet/jetson-nano:applications-smart-distancing" .

# 3) Run Docker container:
docker run -it --runtime nvidia --privileged -p HOST_PORT:8000 -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/jetson-nano:applications-smart-distancing
```

**Run on Coral Dev Board**
```
cd neuralet/applications/smart-distancing/

# 1) Build Docker image (This step is optional, you can skip it if you want to pull the container from neuralet dockerhub)
docker build -f coral-dev-board.Dockerfile -t "neuralet/coral-dev-board:applications-smart-distancing" .
# 2) Run Docker container:
docker run -it --privileged -p HOST_PORT:8000 -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/coral-dev-board:applications-smart-distancing
```

**Run on AMD64 node with a connected Coral USB Accelerator**
```
cd neuralet/applications/smart-distancing/

# 1) Build Docker image (This step is optional, you can skip it if you want to pull the container from neuralet dockerhub)
docker build -f amd64-usbtpu.Dockerfile -t "neuralet/amd64:applications-smart-distancing" .
# 2) Run Docker container:
docker run -it --privileged -p HOST_PORT:8000 -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/amd64:applications-smart-distancing
```

**Run on x86**
```
cd neuralet/applications/smart-distancing/

# 1) Build Docker image (This step is optional, you can skip it if you want to pull the container from neuralet dockerhub)
docker build -f x86.Dockerfile -t "neuralet/x86_64:applications-smart-distancing" .
# 2) Run Docker container:
docker run -it -p HOST_PORT:8000 -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/x86_64:applications-smart-distancing
```

**Run on x86 using OpenVino**


```
cd neuralet/applications/smart-distancing/
# download model first
./download_openvino_model.sh

# 1) Build Docker image (This step is optional, you can skip it if you want to pull the container from neuralet dockerhub)
docker build -f x86-openvino.Dockerfile -t "neuralet/x86_64-openvino:applications-smart-distancing" .
# 2) Run Docker container:
docker run -it -p HOST_PORT:8000 -v /PATH_TO_CLONED_REPO_ROOT/:/repo neuralet/x86_64-openvino:applications-smart-distancing
```

### Configurations
You can read and modify the configurations in `config-jetson.ini` file for Jetson Nano and `config-skeleton.ini` file for Coral.

Under the `[Detector]` section, you can modify the `Min score` parameter to define the person detection threshold. You can also change the distance threshold by altering the value of `DistThreshold`.

## Issues and Contributing

The project is under substantial active development; you can find our roadmap at https://github.com/neuralet/neuralet/projects/1. Feel free to open an issue, send a Pull Request, or reach out if you have any feedback.
* [Submit a feature request](https://github.com/neuralet/neuralet/issues/new?assignees=&labels=&template=feature_request.md&title=).
* If you spot a problem or bug, please let us know by [opening a new issue](https://github.com/neuralet/neuralet/issues/new?assignees=&labels=&template=bug_report.md&title=).


## Contact Us

* Visit our website at https://neuralet.com
* Email us at covid19project@neuralet.com
* Check out our other models at https://github.com/neuralet.
