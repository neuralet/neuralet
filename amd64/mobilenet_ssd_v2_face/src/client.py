from multiprocessing.managers import BaseManager
import cv2 as cv
import numpy as np
import sys

HOST = "127.0.0.1"
INPUT_PORT = 50002
INPUT_AUTH = b"inputpass"
OUTPUT_PORT = 50003
OUTPUT_AUTH = b"outpass"

image_size = [320, 320, 3]


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
img_org = cv.imread(
    img_path
)  # image.load_img(img_path, target_size=image_size[:2])
img_resized = cv.resize(img_org, tuple(image_size[:2]))
img_rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)

for i in range(100):
    input_queue.put(img_rgb)
    detection_result_dict = output_queue.get()
    # get outpu tensor
    boxes = detection_result_dict["boxes"]
    labels = detection_result_dict["labels"]
    scores = detection_result_dict["scores"]
    num = detection_result_dict["num"]

    # visualize:
    # for i in range(boxes.shape[1]):
    #    if scores[0, i] > 0.5:
    #        box = boxes[0, i, :]
    #        x0 = int(box[1] * img_org.shape[1])
    #        y0 = int(box[0] * img_org.shape[0])
    #        x1 = int(box[3] * img_org.shape[1])
    #        y1 = int(box[2] * img_org.shape[0])
    #        box = box.astype(np.int)
    #        cv.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
    #        cv.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
    #        cv.putText(img_org,
    #               str(int(labels[0, i])),
    #               (x0, y0),
    #               cv.FONT_HERSHEY_SIMPLEX,
    #               1,
    #               (255, 255, 255),
    #               2)

    # cv.imwrite('output.jpg', img_org)

    print("got ", num, "boxes")
