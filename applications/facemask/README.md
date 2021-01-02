# Face Mask Detection Application
Face-Mask is an open-source application that can detect and classify faces them to Face and Mask classes. The application is capable to run at x86, Coral Dev Board TPU, and amd64 node with attached usb edge TPU.


## Getting Started
The following instructions will help you get started.

### Prerequisites

#### Hardware
A host edge device. We currently support the following:

* Coral Dev Board
* AMD64 node with attached Coral USB Accelerator
* X86 node
* Jetson 

#### Software

* You should have [Docker](https://docs.docker.com/get-docker/) on your device. 
* Note that you should have Nvidia Docker Toolkit to run the app for training and inference with GPU support.

### Install
Make sure you have the prerequisites and then clone this repository to your local system by running this command:
```
git clone https://github.com/neuralet/neuralet
cd neuralet/applications/facemask/
```
## Usage
The application has a module for trainig Face-Mask classifier and three inference mode:
1- Inference on app mode (using WebGUI to show the result)
2- Inference on video and save output on a video file.
3- Inference on image and save output on a image file.
which are compatible with x86 devices, Coral USB Accelerator, Coral Dev Borad and Jetson.

* NOTE: There is a bunch of config files at `configs/` directory for customizing the parameters of the model and the application. Please set the parameters if you plan to have a customized setting  
* Input ImageSize for Openpifpaf detector should be multiples of 16 plus 1 ( [321,193] and [641,369] is supported on jetson devices for now)

### Run on x86
On x86 devices, you can use two different face detectors. Openpipaf and tiny face detector [[1]](#1).
```
# 1) Build Docker image
docker build -f x86.Dockerfile -t "neuralet/face-mask:latest-x86" .

# 2) Run Docker container:
docker run  -it --gpus all -p HOST_PORT:8000 -v "$PWD/../../":/repo/ -it neuralet/face-mask:latest-x86

# 3) Run main application python script inside the docker 
# config-x86.json: runs openpifpaf as a face detector
# config-tinyface-x86.json: runs tiny face detetcor
python3 inference_main_app.py --config configs/config-x86.json 
```
### Run on AMD64 node with a connected Coral USB Accelerator
```
# 1) Build Docker image
docker build -f amd64-usbtpu.Dockerfile -t "neuralet/face-mask:latest-amd64" .

# 2) Run Docker container:
docker run  -it --privileged -p HOST_PORT:8000 -v "$PWD/../../":/repo/ -it neuralet/face-mask:latest-amd64

# 3) Run main application python script inside the docker
python3 inference_main_app.py --config configs/config_edgetpu.json 
```
### Run on Coral Dev Board
```
# 1) Build Docker image
docker build -f amd64-usbtpu.Dockerfile -t "neuralet/face-mask:latest-coral-dev-board" .

# 2) Run Docker container:
docker run  -it --privileged -p HOST_PORT:8000 -v "$PWD/../../":/repo/ -it neuralet/face-mask:latest-coral-dev-board

# 3) Run main application python script inside the docker
python3 inference_main_app.py --config configs/config_edgetpu.json 
```
### Run on Jetson
You need to have JetPack 4.4 installed on your Jetson.
```
# 1) Build Docker image
docker build -f jetson-4-4.Dockerfile -t "neuralet/face-mask:latest-jetson-4-4" .

# 2) Run Docker container:
docker run  -it --privileged --runtime nvidia -p HOST_PORT:8000 -v "$PWD/../../":/repo/ -it neuralet/face-mask:latest-jetson-4-4

# 3) Run main application python script inside the docker
python3 inference_main_app.py --config configs/config_jetson.json
```

### Train The Model
On X86 devices it is possible to train the face-mask classifier with new dataset, first step for training the model is creating a train and validation set and set like the below structure.
Ex:
```
# train/
# |_face
# |    |__face1.jpg
# |    |__face2.jpg
# |_face-mask
#      |__face-mask1.jpg
#      |__face-mask2.jpg

# validation/
# |_face
# |    |__face1.jpg
# |    |__face2.jpg
# |_face-mask
#      |__face-mask1.jpg
#      |__face-mask2.jpg
```
Set dataset path at `configs/config-x86.json` file according and run the below script.

```
python3 model_main.py --config configs/config-x86.json
```
### Inferencing On Video and Images
There are two easy-to-use scripts for inferencing on video and image.
- The following script get a video file as its input and export the output video at given path. 

`python3 inference_video.py --config configs/config-x86.json --input_video_path data/video/sample.mov --output_video data/videos/output.avi`

- For inferencing on images run the below scrip.

`python3 inference_images.py --config configs/config-x86.json --input_image_dir data/images --output_image_dir output_images`

* NOTE: All scripts should be run inside the docker.


### License
This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. http://creativecommons.org/licenses/by-nc-sa/4.0/ 
If you need to use this code or model for comeercial applications, please reach out to us at hello@neuralet.com

## References
<a id="1">[1]</a>
Hu, Peiyun and Ramanan, Deva, Finding Tiny Faces, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017). [project page](https://www.cs.cmu.edu/~peiyunh/tiny/), [arXiv](https://arxiv.org/abs/1612.04402)
