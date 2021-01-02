import json


class Config:
    """
    Config class load a json file and provide a high-level of abstraction for parameters.
    :param path: the path of json file.
    """
    def __init__(self, path: str):
        self._path = path
        self._config = self._file_loader()

        self.VALID_FOLDER = self._config["validation"]["validation_image_folder"]

        self.MODEL_NAME = self._config["model"]["backend"]
        self.INCLUDE_TOP = self._config["model"]["include_top"]
        self.TRAINING = self._config["model"]["training"]

        self.PATH = self._config["dataset"]["path"]
        self.INPUT_SIZE = self._config["dataset"]["img_size"]
        self.NO_CHANNEL = self._config["dataset"]["no_channel"]
        self.NO_CLASSES = self._config["dataset"]["no_classes"]

        self.TRAIN_FOLDER = self._config["train"]["train_image_folder"]
        self.SAVED_FOLDER = self._config["train"]["saved_weights_folder"]
        self.BATCH_SIZE = self._config["train"]["batch_size"]
        self.LEARNING_RATE = self._config["train"]["learning_rate"]
        self.EPOCHS = self._config["train"]["epochs"]
        self.EXPORT_DIR = self._config["train"]["export_folder"]
        self.LOAD_PRETRAINED = self._config["model"]["load_pretrained"]
        self.PRETRAINED_MODEL = self._config["model"]["pretrained_model"]

        self.CLASSIFIER_NAME = self._config["classifier"]["name"]
        self.CLASSIFIER_MODEL_PATH = self._config["classifier"]["model_path"]
        self.CLASSIFIER_INPUT_SIZE = self._config["classifier"]["input_size"]
        self.CLASSIFIER_TENSORRT_PRECISION = None
        if "tensorrt_precision" in self._config["classifier"]:
            self.CLASSIFIER_TENSORRT_PRECISION = self._config["classifier"]["tensorrt_precision"]
        self.DETECTOR_NAME = self._config["detector"]["name"]
        self.DETECTOR_MODEL_PATH = self._config["detector"]["model_path"]
        self.DETECTOR_INPUT_SIZE = self._config["detector"]["input_size"]
        self.DETECTOR_TENSORRT_PRECISION = None
        if "tensorrt_precision" in self._config["detector"]:
            self.DETECTOR_TENSORRT_PRECISION = self._config["detector"]["tensorrt_precision"]
        self.DEVICE = self._config["app"]["device"]
        self.APP_VIDEO_PATH = self._config["app"]["video_path"]
        self.APP_VIDEO_RESOLUTION = self._config["app"]["resolution"]
        self.APP_HOST = self._config["app"]["host"]
        self.APP_PORT = self._config["app"]["port"]
        
    def _file_loader(self) -> dict:
        """
        Load json file which is path is set at self.path
        :return: cfg
        """
        cfg = None
        try:
            with open(self._path) as file:
                cfg = json.loads(file.read())
        except FileNotFoundError as e:
            print(e)
            exit(1)
        return cfg
