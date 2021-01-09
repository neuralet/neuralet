# Adaptive Learning  of Object Detection Models


The goal of adaptive learning is to build robust systems that can work efficiently and adapt to new environments (novel data distributions) without labeled data or manual labeling or storing or transferring data from the edge devices to other systems. This framework will dramatically simplify machine learning systems’ architecture and lower their deployment and maintenance costs when applied.
We published a comprehensive blog about the concept and features of Adaptive Learning. Please refer to the [blog post](https://neuralet.com/article/adaptive-learning-part-1/) for more details.

## prerequisites
1. At least one Nvidia GPU
2. At least 6GB of RAM
3. [Docker](https://docs.docker.com/get-docker/)
4. [Docker Compose](https://github.com/docker/compose)
5. [Nvidia Docker Toolkit](https://github.com/NVIDIA/nvidia-docker)

## Getting Started
### Clone Neuralet Repository
Make sure you have the prerequisites and then clone this repository to your local system by running this command:
```
git clone https://github.com/neuralet/neuralet.git

cd neuralet/applications/adaptive-learning
```
### Download Sample Video (In Case of Trying Demo):

```
bash ./download_sample_video.sh
```
### Configure Adaptive Learning Config File
To customize the Adaptive Learning framework based on your needs, you must configure one of the sample config files on `configs/` directory.
there is a brief explanation on each parameter of config files in the following table:


| Parameter    | Options | Comments                                                                                                                          |
| ------------ | ------- | -------- |
| Teacher/Name     | iterdet/faster_rcnn_nas     | name of the teacher model. (in case of implementation of new teacher you should register its name in `teachers/model_builder.py`)     |
| Teacher/ImageSize | 300,300,3 (or any other appropriate image size) | the input image size of the teacher model |
| Teacher/ClassID | integer | The pedestrian class ID of the Teacher. (this parameter is important only where the teacher model is a multiple class object detector.) |
| Teacher/MinScore | a float number between 0 and 1 | the teacher model threshold for detecting an object |
| Teacher/VideoPath | a unix-like path | path to the video file that you want to apply adaptive learning. (don't change it if you just want run demo)  |
| Teacher/MaxAllowedImage | an integer number | Maximum number of images that can be stored in hard, when this threshold is exeeded, the teacher model stops processing video frames until the student models picks some of the images and remove them from hard disk. |
| Teacher/MinDetectionPerFrame | an integer number | The teacher only stores the frame only if it detects at least this many objects, otherwise it discards the frame.|
| Teacher/SaveFrequency | An integer bigger than 0 | The teacher model will store video frames and the corresponding predictions with this frequency |
| Teacher/PostProcessing | one of `"background_filter"` or `" "` | Background filter will apply a background subtraction algorithm on video frames and discards the bounding boxes in which their background pixels rate is higher than a defined threshold. |
| Teacher/ImageFeature | One of the `"foreground_mask"`, `"optical_flow_magnitude"`, `"foreground_mask && optical_flow_magnitude"` or `" "` |This parameter specifies the type of input feature engineering that will perform for training. `"foreground_mask"` replaces one of the RGB channels with the foreground mask. `"optical_flow_magnitude"` replaces one of the RGB channels with the magnitude of optical flow vectors and, `"foreground_mask && optical_flow_magnitude"` performs two feature engineering technique at the same time as well as changing the remaining RGB channel with the grayscale transformation of the frame. For more information about feature engineering and their impact on the model's accuracy, visit our blog.
| Student/BatchSize | An integer bigger than 0 | The student's training batch size |
| Student/Epochs | An integer bigger than 0 | The student's training number of epochs in each round |
| Student/ValidationSplit | An float between 0 and 1 | the portions of data which will be used for validation in each training round |
| Student/ExamplePerRound | An integer bigger than 0 | How many example to use for each training round? |
| Student/TrainingRounds | An integer bigger than 0 | how many rounds you want to run Adaptive Learning? |
| Student/QuantizationAware | true or false | whether to train the student model with quantization aware strategy or not. This is specially useful when you want to deploy the final model on a edge device that only supports `Int8` precision like Edge TPU. By applying quantization aware training the student model will be exported to `tflite` too.|


### Running the Docker Compose

After configuring the config file, you just need to run following command to start the Adaptive Learning:
```
# For running with IterDet teacher:
docker-compose -f docker-compose-iterdet.yml up

# For running with Faster RCNN NAS teacher:
docker-compose -f docker-compose-faster-rcnn-nas.yml up
```
After the training rounds completed you can find the final student model under: `data/student_model/frozen_graph/`

### Tracking Training Process
In the `docker-compose.yml` file, you can forward the 6006 port (tensorboard port) of the student's container to one of your machine's open ports (default port is 2029). then browse this port in the browser to track and monitor training with tensorboard.

### Defining Your Own Teacher Model.
You can add your own teacher model to the Adaptive Learning framework by writing a class which is a subclass of `TeacherMetaArcht` class and implement `inference` method. please check out the `teachers/teacher_meta_arch.py` and `teachers/iterdet.py` for more information. Note that you should register your model in `teachers/model_builder.py` too.
