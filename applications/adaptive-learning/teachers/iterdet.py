import os
import logging
import wget
from mmdet.apis import init_detector, inference_detector
from teacher_meta_arch import TeacherMetaArch
import torch


class IterDet(TeacherMetaArch):
    """ IterDet object detection model"""

    def __init__(self, config):
        """
        IterDet class constructor
        Args:
            config: an adaptive learning config file
        """
        super(IterDet, self).__init__(config=config)
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
        output_dict = inference_detector(self.detection_model, preprocessed_image)

        result = []
        for i, box in enumerate(output_dict[0]):  # number of boxes
            if box[-1] > self.score_threshold:
                result.append({"id": str(self.class_id) + '-' + str(i),
                               "bbox": [box[0] / self.image_size[0], box[1] / self.image_size[1],
                                        box[2] / self.image_size[0], box[3] / self.image_size[1]],
                               "score": box[-1]})

        return result

    def load_model(self):
        """
        This function will load the IterDet model with its checkpoints on Crowd Human dataset. The chechpoints and configs
        will be download in "data/iterdet" directory if they do not exists already.
        """
        base_path = "data/iterdet"
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        config_file = os.path.join(base_path, "crowd_human_full_faster_rcnn_r50_fpn_2x.py")
        if not os.path.isfile(config_file):
            url = "https://raw.githubusercontent.com/saic-vul/iterdet/master/" \
                  "configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py"
            logging.info(f'config file does not exist under: {config_file}, downloading from {url}')
            wget.download(url, config_file)

        checkpoint_file = os.path.join(base_path, "crowd_human_full_faster_rcnn_r50_fpn_2x.pth")
        if not os.path.isfile(checkpoint_file):
            url = "https://github.com/saic-vul/iterdet/releases/download/v2.0.0/crowd_human_full_faster_rcnn_r50_fpn_2x.pth"
            logging.info(f'checkpoint file does not exist under: {checkpoint_file}, downloading from {url}')
            wget.download(url, checkpoint_file)

        # build the model from a config file and a checkpoint file
        device = self.config.get_section_dict('Teacher')['Device']
        if device == "GPU" and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        model = init_detector(config_file, checkpoint_file, device=device)

        return model
