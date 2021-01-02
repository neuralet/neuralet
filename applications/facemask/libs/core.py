import cv2 as cv
import numpy as np


class FaceMaskAppEngine:
    """
    Perform detector which detects faces from input video,
    and classifier to classify croped faces to face or mask class
    :param config: Is a Config instance which provides necessary parameters.
    """

    def __init__(self, config):
        self.config = config
        self.detector = None
        self.classifier_model = None
        self.running_video = False
        self.device = self.config.DEVICE
        if self.device == "x86":
            from libs.detectors.x86.detector import Detector
            from libs.classifiers.x86.classifier import Classifier
            self.detector = Detector(self.config)
            self.classifier_model = Classifier(self.config)
        elif self.device == "EdgeTPU":
            from libs.detectors.edgetpu.detector import Detector
            from libs.classifiers.edgetpu.classifier import Classifier
            self.detector = Detector(self.config)
            self.classifier_model = Classifier(self.config)
        elif self.device == "Jetson":
            from libs.detectors.jetson.detector import Detector
            from libs.classifiers.jetson.classifier import Classifier
            self.detector = Detector(self.config)
            self.classifier_model = Classifier(self.config)
        else:
            raise ValueError('Not supported device named: ', self.device)

        self.image_size = (self.config.DETECTOR_INPUT_SIZE[0], self.config.DETECTOR_INPUT_SIZE[1], 3)
        self.classifier_img_size = (self.config.CLASSIFIER_INPUT_SIZE, self.config.CLASSIFIER_INPUT_SIZE, 3)

    def set_ui(self, ui):
        self.ui = ui

    def __process(self, cv_image):
        # Resize input image to resolution
        self.resolution = self.config.APP_VIDEO_RESOLUTION
        cv_image = cv.resize(cv_image, tuple(self.resolution))

        resized_image = cv.resize(cv_image, tuple(self.image_size[:2]))
        rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
        objects_list = self.detector.inference(rgb_resized_image)
        [w, h] = self.resolution
        #objects_list = [{'id': '1-0', 'bbox': [.1, .2, .5, .5]}, {'id': '1-1', 'bbox': [.3, .1, .5, .5]}]
        faces = []
        for obj in objects_list:
            if 'bbox' in obj.keys():
                face_bbox = obj['bbox']  # [ymin, xmin, ymax, xmax]
                xmin, xmax = np.multiply([face_bbox[1], face_bbox[3]], self.resolution[0])
                ymin, ymax = np.multiply([face_bbox[0], face_bbox[2]], self.resolution[1])
                croped_face = cv_image[int(ymin):int(ymin) + (int(ymax) - int(ymin)),
                                      int(xmin):int(xmin) + (int(xmax) - int(xmin))]
                # Resizing input image
                croped_face = cv.resize(croped_face, tuple(self.classifier_img_size[:2]))
                croped_face = cv.cvtColor(croped_face, cv.COLOR_BGR2RGB)
                # Normalizing input image to [0.0-1.0]
                croped_face = croped_face / 255.0
                faces.append(croped_face)
        
        faces = np.array(faces)
        face_mask_results, scores = self.classifier_model.inference(faces)

        # TODO: it could be optimized by the returned dictionary from openpifpaf (returining List instead dict)
        [w, h] = self.resolution
        
        idx = 0
        for obj in objects_list:
            if 'bbox' in obj.keys():
                obj['face_label'] = face_mask_results[idx] 
                obj['score'] = scores[idx]
                idx = idx + 1
                box = obj["bbox"]
                x0 = box[1]
                y0 = box[0]
                x1 = box[3]
                y1 = box[2]
                obj["bbox"] = [x0, y0, x1, y1]


        return cv_image, objects_list

    def process_video(self, video_uri):
        input_cap = cv.VideoCapture(video_uri)

        if (input_cap.isOpened()):
            print('opened video ', video_uri)
        else:
            print('failed to load video ', video_uri)
            return

        self.running_video = True
        while input_cap.isOpened() and self.running_video:
            _, cv_image = input_cap.read()
            if np.shape(cv_image) != ():
                cv_image, objects = self.__process(cv_image)
            else:
                continue
            self.ui.update(cv_image, objects)
        input_cap.release()
        self.running_video = False

    # def process_image(self, image_path):
    #     # Process and pass the image to ui modules
    #     cv_image = cv.imread(image_path)
    #     cv_image, objects, distancings = self.__process(cv_image)
    #     self.ui.update(cv_image, objects, distancings)
