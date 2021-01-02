from configs.config_handler import Config
import cv2 as cv
import numpy as np
from argparse import ArgumentParser
import os


def main():
    """
    Read input images and process them, the output images will be exported output_dir
     which can be set by input arguments.
    Example: python inference_images.py --config configs/config.json --input_image_dir data/images
     --output_image_dir output_images
    """
    argparse = ArgumentParser()
    argparse.add_argument('--config', type=str, help='json config file path')
    argparse.add_argument('--input_image_dir', type=str, help='the directory of input images')
    argparse.add_argument('--output_image_dir', type=str, help='the directory of output images',
                          default='output_images')
    args = argparse.parse_args()

    config_path = args.config
    cfg = Config(path=config_path)

    input_dir = args.input_image_dir
    print("INFO: The directory of input images is: ", input_dir)
    output_dir = args.output_image_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.isdir(output_dir):
        print('"{} "output directory is not exists please make the directory before running this script.'.format(
            output_dir))
        exit(1)

    print("INFO: The output images will be exported at: ", output_dir)

    detector_input_size = (cfg.DETECTOR_INPUT_SIZE[0], cfg.DETECTOR_INPUT_SIZE[1], 3)
    classifier_img_size = (cfg.CLASSIFIER_INPUT_SIZE, cfg.CLASSIFIER_INPUT_SIZE, 3)

    device = cfg.DEVICE
    detector = None
    classifier = None

    if device == "x86":
        from libs.detectors.x86.detector import Detector
        from libs.classifiers.x86.classifier import Classifier

    elif device == "EdgeTPU":
        from libs.detectors.edgetpu.detector import Detector
        from libs.classifiers.edgetpu.classifier import Classifier
    elif device == "Jetson":
        from libs.detectors.jetson.detector import Detector
        from libs.classifiers.jetson.classifier import Classifier
    else:
        raise ValueError('Not supported device named: ', device)

    detector = Detector(cfg)
    classifier_model = Classifier(cfg)

    print("INFO: Start inferencing")
    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        raw_img = cv.imread(image_path)
        if np.shape(raw_img) != ():
            height, width, _ = raw_img.shape
            resized_image = cv.resize(raw_img, tuple(detector_input_size[:2]))
            rgb_resized_image = cv.cvtColor(resized_image, cv.COLOR_BGR2RGB)
            objects_list = detector.inference(rgb_resized_image)
            faces = []
            cordinates = []
            cordinates_head = []
            for obj in objects_list:
                if 'bbox' in obj.keys():
                    face_bbox = obj['bbox']  # [ymin, xmin, ymax, xmax]
                    xmin, xmax = np.multiply([face_bbox[1], face_bbox[3]], width)
                    ymin, ymax = np.multiply([face_bbox[0], face_bbox[2]], height)
                    croped_face = raw_img[int(ymin):int(ymin) + (int(ymax) - int(ymin)),
                                  int(xmin):int(xmin) + (int(xmax) - int(xmin))]
                    # Resizing input image
                    croped_face = cv.resize(croped_face, tuple(classifier_img_size[:2]))
                    croped_face = cv.cvtColor(croped_face, cv.COLOR_BGR2RGB)
                    # Normalizing input image to [0.0-1.0]
                    croped_face = croped_face / 255.0
                    faces.append(croped_face)
                    cordinates.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                if 'bbox_head' in obj.keys():
                    head_bbox = obj['bbox_head']  # [ymin, xmin, ymax, xmax]
                    xmin, xmax = np.multiply([head_bbox[1], head_bbox[3]], width)
                    ymin, ymax = np.multiply([head_bbox[0], head_bbox[2]], height)
                    cordinates_head.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            
            faces = np.array(faces)
            if np.shape(faces)[0] == 0:
                print("can not find face at ".image_path)
                continue

            face_mask_results, scores = classifier_model.inference(faces)
            for i, cor in enumerate(cordinates):
                if face_mask_results[i] == 1:
                    color = (0, 0, 255)
                elif face_mask_results[i] == 0:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 0)

                cv.rectangle(raw_img, (cor[0], cor[1]), (cor[2], cor[3]), color, 2)
            for cor in cordinates_head:
                cv.rectangle(raw_img, (cor[0], cor[1]), (cor[2], cor[3]), (200,200,200), 2)
            cv.imwrite(os.path.join(output_dir, filename), raw_img)
            print('image "{}" are processed. {} fps'.format(image_path, detector.fps))
        else:
            continue

    print('INFO: Finish:) Output images are exported at: ', output_dir)


if __name__ == '__main__':
    main()
