import os
import collections
import numpy as np
import time
from pose_engine import PoseEngine
from libs.utils.fps_calculator import convert_infr_time_to_fps


class Detector:
    """
    Perform pose estimation with Openpifpaf model. extract pedestrian's bounding boxes from key-points.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        # Get the model name from the config
        self.model_name = self.config.DETECTOR_NAME
        # Frames Per Second
        self.fps = None 
        self.engine = PoseEngine(
                "/project-posenet/models/mobilenet/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite"
                )

        self.w, self.h, = self.config.DETECTOR_INPUT_SIZE[0], self.config.DETECTOR_INPUT_SIZE[1]
        self.keypoints = (
	      'nose',
	      'left eye',
	      'right eye',
	      'left ear',
	      'right ear',
	      'left shoulder',
	      'right shoulder',
	      'left elbow',
	      'right elbow',
	      'left wrist',
	      'right wrist',
	      'left hip',
	      'right hip',
	      'left knee',
	      'right knee',
	      'left ankle',
	      'right ankle'
	    )


    def inference(self, resized_rgb_image):
        """
        This method will perform inference and return the detected bounding boxes
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)
        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        assert resized_rgb_image.shape == (721, 1281, 3)
        t_begin = time.perf_counter()
        poses, _ = self.engine.DetectPosesInImage(resized_rgb_image)
        inference_time = time.perf_counter() - t_begin  # Second
        self.fps = convert_infr_time_to_fps(inference_time)
        result = []
        for i, pose in enumerate(poses):  # number of boxes
            if pose.score > 0.25:
                pose_dict = collections.OrderedDict(sorted(pose.keypoints.items(), key=lambda x: self.keypoints.index(x[0])))
                keypoints = np.array(
                    [[keypoint.yx[1], keypoint.yx[0], keypoint.score] for _, keypoint in pose_dict.items()])
                bbox_dict = {}
                if np.all(keypoints[[0, 1, 2, 5, 6], -1] > 0.15):
                    x_min_face = int(keypoints[6, 0]) / self.w
                    x_max_face = int(keypoints[5, 0]) / self.w
                    y_max_face = int((keypoints[5, 1] + keypoints[6, 1]) / 2) /self.h
                    y_eyes = int((keypoints[1, 1] + keypoints[2, 1]) / 2) / self.h
                    y_min_face = 2 * y_eyes - y_max_face
                    if (y_max_face - y_min_face > 0) and (x_max_face - x_min_face > 0):
                        h_crop = y_max_face - y_min_face
                        x_min_face = max(0, x_min_face - 0.2 * h_crop)
                        y_min_face = max(0, y_min_face - 0.0 * h_crop)
                        x_max_face = min(1, x_min_face + 1.0 * h_crop)
                        y_max_face = min(1, y_min_face + 1.0 * h_crop)
                        bbox_dict["bbox"] = [y_min_face, x_min_face, y_max_face, x_max_face]
                else: 
                    x_min_head = self.w
                    y_min_head = self.h
                    x_max_head = 0
                    y_max_head = 0
                    for i in range(6):
                        if keypoints[i,0] > 0.0:
                            x_min_head = min(x_min_head, keypoints[i,0])
                        if keypoints[i,1] > 0.0:
                            y_min_head = min(y_min_head, keypoints[i,1])
                        x_max_head = max(x_max_head, keypoints[i,0])
                        y_max_head = max(y_max_head, keypoints[i,1])
                        h_crop = y_max_head - y_min_head
                    if ( x_min_head != self.w and x_max_head != 0 and y_min_head != self.h and y_max_head != 0 and x_min_head != x_max_head and y_min_head != y_max_head ):
                        x_min_head = int(max(0, x_min_head - 0.2 * h_crop))
                        y_min_head = int(max(0, y_min_head - 0.4 * h_crop))
                        x_max_head = int(min(self.w, x_max_head + 1 * h_crop))
                        y_max_head = int(min(self.h, y_max_head + 0.8 * h_crop))

                        bbox_dict["bbox_head"] = [y_min_head / self.h, x_min_head / self.w, y_max_head / self.h, x_max_head / self.w]
     
                result.append(bbox_dict)

        return result
