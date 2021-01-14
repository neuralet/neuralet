import sys
import numpy as np
from multiprocessing.managers import BaseManager

import rospy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv

HOST = "127.0.0.1"
INPUT_PORT = 50002
INPUT_AUTH = b"inputpass"
OUTPUT_PORT = 50003
OUTPUT_AUTH = b"outpass"

DEFAULT_IMAGE_TOPIC = "image"
DEFAULT_RESULT_TOPIC = "result"
DEFAULT_TOPICS_QUEUE_SIZE = 10

IMAGE_SIZE = [240, 240, 3]


class QueueManager(BaseManager):
    pass


QueueManager.register("get_input_queue")
QueueManager.register("get_output_queue")


class NeuraletBridge:
    def __init__(self):
        self.input_manager = QueueManager(
            address=(HOST, INPUT_PORT), authkey=INPUT_AUTH
        )
        self.input_manager.connect()

        self.output_manager = QueueManager(
            address=(HOST, OUTPUT_PORT), authkey=OUTPUT_AUTH
        )
        self.output_manager.connect()
        self.cv_bridge = CvBridge()

        self.input_queue = self.input_manager.get_input_queue()
        self.output_queue = self.output_manager.get_output_queue()

        self.image_subscriber = rospy.Subscriber(
            rospy.get_param("~image", DEFAULT_IMAGE_TOPIC),
            Image,
            self.bridge_callback,
            queue_size=rospy.get_param("~qsize", DEFAULT_TOPICS_QUEUE_SIZE),
        )
        self.result_publisher = rospy.Publisher(
            rospy.get_param("~result", DEFAULT_RESULT_TOPIC),
            Int32,
            queue_size=rospy.get_param("~qsize", DEFAULT_TOPICS_QUEUE_SIZE),
        )

    def bridge_callback(self, image_message):
        img_rgb = self.cv_bridge.imgmsg_to_cv2(image_message, "rgb8")
        img_rgb_resized = cv.resize(img_rgb, tuple(IMAGE_SIZE[:2]))
        self.input_queue.put(img_rgb_resized.tolist())
        net_output = self.output_queue.get()
        # decode your output here ...
        class_id = np.argmax(net_output)
        # publish the result ...
        self.result_publisher.publish(class_id)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    # initiate ros node:
    rospy.init_node("neuralet_bridge")
    # create and run the bridge:
    neuralet_bridge = NeuraletBridge()
    try:
        print "bringing up neuralet bridge ..."
        neuralet_bridge.run()
    except rospy.ROSInterruptException:
        pass
