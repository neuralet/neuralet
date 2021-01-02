import numpy as np 
import pickle 

import openpifpaf.decoder.cifcaf as OriginalDecoder

from openpifpaf import visualizer
from openpifpaf.datasets.constants import COCO_KEYPOINTS, COCO_PERSON_SKELETON

class CifCafDecoder():
    def __init__(self):
        self.cif_metas = pickle.load(open( "/repo/applications/facemask/libs/detectors/jetson/openpifpaf_tensorrt/cif_metas.pkl", "rb" ))
        self.caf_metas = pickle.load(open( "/repo/applications/facemask/libs/detectors/jetson/openpifpaf_tensorrt/caf_metas.pkl", "rb" ))
        self._decoder = OriginalDecoder.CifCaf(cif_metas = self.cif_metas, 
                                                caf_metas = self.caf_metas) 

    def decode(self, fields):
        return self._decoder(fields)

