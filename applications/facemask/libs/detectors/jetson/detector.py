class Detector:
    """
    Detector class is a high level class for detecting object using NVIDIA jetson devices.
    When an instance of the Detector is created you can call inference method and feed your
    input image in order to get the detection results.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        self.net = None
        self.fps = None
        # Get model name from the config
        self.name = self.config.DETECTOR_NAME
        if self.name == "openpifpaf_tensorrt":
            from libs.detectors.jetson.openpifpaf_tensorrt import openpifpaf_tensorrt
            self.net = openpifpaf_tensorrt.Detector(self.config)

        else:
            raise ValueError('Not supported network named: ', self.name)
    
    def __del__(self):
        del self.net

    def inference(self, resized_rgb_image):
        """
        Run inference on an image and get Frames rate (fps)

        Args:
            resized_rgb_image: A numpy array with shape [height, width, channels]

        Returns:
            output: List of objects, each obj is a dict with two keys "id" and "bbox" and "score"
            e.g. [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        self.fps = self.net.fps
        output = self.net.inference(resized_rgb_image)
        return output
