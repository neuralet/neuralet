from multiprocessing.managers import BaseManager
import cv2 as cv
import numpy as np
import sys

HOST = "127.0.0.1"
INPUT_PORT = 50002
INPUT_AUTH = b"inputpass"
OUTPUT_PORT = 50003
OUTPUT_AUTH = b"outpass"

image_size = [224, 224, 3]


class QueueManager(BaseManager):
    pass


QueueManager.register("get_input_queue")
input_manager = QueueManager(address=(HOST, INPUT_PORT), authkey=INPUT_AUTH)
input_manager.connect()

QueueManager.register("get_output_queue")
output_manager = QueueManager(address=(HOST, OUTPUT_PORT), authkey=OUTPUT_AUTH)
output_manager.connect()

input_queue = input_manager.get_input_queue()
output_queue = output_manager.get_output_queue()

# Optional image to test model prediction.
import sys

img_path = sys.argv[1]  # or 'elephant.jpg'
if img_path == "stop":
    input_queue.put("stop")
    exit(0)

# prepare your input
img = cv.imread(
    img_path
)  # image.load_img(img_path, target_size=image_size[:2])
img = cv.resize(img, tuple(image_size[:2]))
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

for i in range(100):
    input_queue.put(img_rgb)
    print(np.argmax(output_queue.get()))
