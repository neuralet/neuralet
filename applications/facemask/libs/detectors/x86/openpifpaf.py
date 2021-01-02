import openpifpaf
import torch
import numpy as np
import time
from libs.utils.fps_calculator import convert_infr_time_to_fps
import PIL


class Detector:
    """
    Perform pose estimation with Openpifpaf model. extract pedestrian's bounding boxes from key-points.
    :param config: Is a Config instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        # Get the model name from the config
        self.model_name = self.config.DETECTOR_NAME
        # Frames Per Second
        self.fps = None
        self.net, self.processor = self.load_model()
        self.w, self.h, = self.config.DETECTOR_INPUT_SIZE[0], self.config.DETECTOR_INPUT_SIZE[1]

    def load_model(self):
        self.device = "cpu"
        net_cpu, _ = openpifpaf.network.factory(checkpoint="resnet50", download_progress=False)
        net = net_cpu.to(self.device)
        openpifpaf.decoder.CifSeeds.threshold = 0.5
        openpifpaf.decoder.nms.Keypoints.keypoint_threshold = 0.2
        openpifpaf.decoder.nms.Keypoints.instance_threshold = 0.2
        processor = openpifpaf.decoder.factory_decode(net.head_nets, basenet_stride=net.base_net.stride)
        return net, processor

    def inference(self, resized_rgb_image):
        """
        This method will perform inference and return the detected bounding boxes
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)
        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]
        """
        pil_im = PIL.Image.fromarray(resized_rgb_image)
        preprocess = openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.CenterPadTight(16),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])
        data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)
        loader = torch.utils.data.DataLoader(
            data, batch_size=1, pin_memory=True,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)
        t_begin = time.perf_counter()
        for images_batch, _, __ in loader:
            predictions = self.processor.batch(self.net, images_batch, device=self.device)[0]
        inference_time = time.perf_counter() - t_begin
        self.fps = convert_infr_time_to_fps(inference_time)
        result = []
        for i, pred in enumerate(predictions):
            pred = pred.data
            pred_visible = pred[pred[:, 2] > .2]
            xs = pred_visible[:, 0]
            ys = pred_visible[:, 1]
            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())
            w = x_max - x_min
            h = y_max - y_min
            bbox_dict = {}
            # extracting face bounding box
            if np.all(pred[[0, 1, 2, 5, 6], -1] > 0.15):
                x_min_face = int(pred[6, 0])
                x_max_face = int(pred[5, 0])
                y_max_face = int((pred[5, 1] + pred[6, 1]) / 2)
                y_eyes = int((pred[1, 1] + pred[2, 1]) / 2)
                y_min_face = 2 * y_eyes - y_max_face
                if (y_max_face - y_min_face > 0) and (x_max_face - x_min_face > 0):
                    h_crop = y_max_face - y_min_face
                    x_min_face = int(max(0, x_min_face - 0.2 * h_crop))
                    y_min_face = int(max(0, y_min_face - 0.1 * h_crop))
                    x_max_face = int(min(self.w, x_min_face + 1 * h_crop))
                    y_max_face = int(min(self.h, y_min_face + 1 * h_crop))
                    bbox_dict["bbox"] = [y_min_face / self.h, x_min_face / self.w, y_max_face / self.h, x_max_face / self.w]
            else: 
                x_min_head = self.w
                y_min_head = self.h
                x_max_head = 0
                y_max_head = 0
                for i in range(6):
                    if pred[i,0] > 0.0:
                        x_min_head = min(x_min_head, pred[i,0])
                    if pred[i,1] > 0.0:
                        y_min_head = min(y_min_head, pred[i,1])
                    x_max_head = max(x_max_head, pred[i,0])
                    y_max_head = max(y_max_head, pred[i,1])
                    h_crop = y_max_head - y_min_head
                if ( x_min_head != self.w and x_max_head != 0 and y_min_head != self.h and y_max_head != 0 and x_min_head != x_max_head and y_min_head != y_max_head ):
                    x_min_head = int(max(0, x_min_head - 0.2 * h_crop))
                    y_min_head = int(max(0, y_min_head - 0.4 * h_crop))
                    x_max_head = int(min(self.w, x_max_head + 1 * h_crop))
                    y_max_head = int(min(self.h, y_max_head + 0.8 * h_crop))

                    bbox_dict["bbox_head"] = [y_min_head / self.h, x_min_head / self.w, y_max_head / self.h, x_max_head / self.w]
 
            result.append(bbox_dict)

        return result
