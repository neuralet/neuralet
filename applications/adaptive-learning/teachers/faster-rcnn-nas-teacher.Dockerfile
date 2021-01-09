FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3

VOLUME  /repo

WORKDIR /repo/applications/adaptive-learning/teachers

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip setuptools==41.0.0 && pip install opencv-python wget scipy image lxml

CMD python teacher_main.py --config $ADAPTIVE_LEARNING_CONFIG
