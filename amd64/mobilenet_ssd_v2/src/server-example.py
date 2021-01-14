import time
from multiprocessing.managers import BaseManager
from queue import Queue
import numpy as np
import sys
import os
import wget

from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

HOST = "127.0.0.1"
INPUT_PORT = 50002
INPUT_AUTH = b"inputpass"
OUTPUT_PORT = 50003
OUTPUT_AUTH = b"outpass"
input_queue = Queue()
output_queue = Queue()


class QueueManager(BaseManager):
    pass


QueueManager.register("get_input_queue", callable=lambda: input_queue)
input_manager = QueueManager(address=(HOST, INPUT_PORT), authkey=INPUT_AUTH)
input_manager.start()

QueueManager.register("get_output_queue", callable=lambda: output_queue)
output_manager = QueueManager(address=(HOST, OUTPUT_PORT), authkey=OUTPUT_AUTH)
output_manager.start()

model_file = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
model_name = "mobilenet_ssd_v2"
model_path = "data/models/" + model_name + "/" + model_file

base_url = "https://raw.githubusercontent.com/neuralet/neuralet-models/master/edge-tpu/"
url = base_url + model_name + "/" + model_file

if not os.path.isfile(model_path):
    print("model does not exist, downloading from ", url)
    wget.download(url, model_path)


def main():

    interpreter = Interpreter(
        model_path, experimental_delegates=[load_delegate("libedgetpu.so.1")]
    )

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_queue = input_manager.get_input_queue()
    output_queue = output_manager.get_output_queue()

    print(
        "------------------------------------------------------------------------------------------"
    )
    print(
        "Started Inference server, waiting for incoming requests ... (send 'stop' to kill server)"
    )
    print(
        "------------------------------------------------------------------------------------------"
    )

    while True:
        data = input_queue.get()
        print("recieved data with type ", type(data))

        if type(data) == str and data == "stop":
            break

        if type(data) == np.ndarray:

            input_image = np.expand_dims(data, axis=0)
            interpreter.set_tensor(input_details[0]["index"], input_image)
            t_begin = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - t_begin
            boxes = interpreter.get_tensor(output_details[0]["index"])
            labels = interpreter.get_tensor(output_details[1]["index"])
            scores = interpreter.get_tensor(output_details[2]["index"])
            num = interpreter.get_tensor(output_details[3]["index"])
            print(
                "number of detected objects: ",
                num,
                ", done in ",
                inference_time,
                " seconds",
            )
            output_queue.put(
                {
                    "num": num,
                    "boxes": boxes,
                    "labels": labels,
                    "scores": scores,
                }
            )

    # End while

    input_manager.shutdown()
    output_manager.shutdown()


if __name__ == "__main__":
    main()
