import pathlib
import os
import numpy as np
import tarfile
import wget
import tensorflow as tf
from teacher_meta_arch import TeacherMetaArch


class FasterRcnnNas(TeacherMetaArch):
    """ Faster RCNN NAS object detection model"""

    def __init__(self, config):
        """
        IterDet class constructor
        Args:
            config: an adaptive learning config file
        """
        super(FasterRcnnNas, self).__init__(config=config)
        self.detection_model = self.load_model()

    def inference(self, preprocessed_image):
        """
        predict bounding boxes of a preprocessed image
        Args:
            preprocessed_image: a preprocessed RGB frame

        Returns:
            A list of dictionaries, each item has the id, relative bounding box coordinate and prediction confidence score
             of one detected instance.
        """
        self.frame = preprocessed_image
        input_image = np.expand_dims(preprocessed_image, axis=0)
        input_tensor = tf.convert_to_tensor(input_image)
        output_dict = self.detection_model(input_tensor)
        boxes = output_dict['detection_boxes']
        labels = output_dict['detection_classes']
        scores = output_dict['detection_scores']
        result = []
        for i in range(boxes.shape[1]):  # number of boxes
            if labels[0, i] == self.class_id and scores[0, i] > self.score_threshold:
                result.append({"id": str(self.class_id) + '-' + str(i), "bbox": self.change_coordinate_order(boxes[0, i, :].numpy().tolist()), "score": scores[0, i]})

        return result

    @staticmethod
    def change_coordinate_order(bbox):
        bbox_new = [bbox[1], bbox[0], bbox[3], bbox[2]]
        return bbox_new

    def load_model(self):
        """
        this method loads Faster RCNN NAS model, The checkpoints and model file will be
        download in data/faster_rcnn_nas directory if they do not exists already.

        Returns:
            A TensorFlow model
        """
        base_url = "http://download.tensorflow.org/models/object_detection/"
        model_file = "faster_rcnn_nas_coco_2018_01_28" + ".tar.gz"
        base_dir = "data/faster_rcnn_nas/"
        model_dir = os.path.join(base_dir, "faster_rcnn_nas_coco_2018_01_28")
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.isfile(os.path.join(base_dir, model_file)):
            print('model does not exist under: ', base_dir, 'downloading from ', base_url + model_file)
            wget.download(base_url + model_file, base_dir)
            with tarfile.open(base_dir + model_file, "r") as tar:
                tar.extractall(path=base_dir)

        model_dir = pathlib.Path(model_dir) / "saved_model"

        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']

        return model
