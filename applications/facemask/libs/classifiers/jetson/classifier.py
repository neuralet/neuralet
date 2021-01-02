import logging
logging.getLogger().setLevel(logging.INFO)

class Classifier:
    """
    Classifier class is a high level class for classifying images using x86 devices.
    When an instance of the Classifier is created you can call inference method and feed your
    input image in order to get the classifier results.

    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """
    def __init__(self, config):
        self.config = config
        self.name = self.config.CLASSIFIER_NAME

        if self.name == 'OFMClassifier':
            from libs.classifiers.jetson import face_mask
            self.net = face_mask.Classifier(self.config)
        else:
            raise ValueError('Not supported network named: ', self.name)

    def inference(self, resized_rgb_image):
        self.fps = self.net.fps
        output, scores = self.net.inference(resized_rgb_image)
        return output, scores
