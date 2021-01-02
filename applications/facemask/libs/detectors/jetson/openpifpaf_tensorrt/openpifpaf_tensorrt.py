import time
import os
import torch
import numpy as np
import cv2
import openpifpaf
import pycuda.driver as cuda
import PIL

import tensorrt as trt
from libs.detectors.jetson.openpifpaf_tensorrt.decoder import CifCafDecoder
from libs.utils.fps_calculator import convert_infr_time_to_fps


def allocate_buffers(engine):
    host_inputs = []
    cuda_inputs = []
    host_outputs = [] 
    cuda_outputs = []
    bindings = [] 
    for i in range(engine.num_bindings):
        binding = engine[i]
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        host_mem = cuda.pagelocked_empty(size, np.float32)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(cuda_mem))
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    stream = cuda.Stream()  # create a CUDA stream to run inference        
    return bindings, host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream

class Detector:
    """
    TODO: UPDATE for TensorRT
    Perform pose estimation with Openpifpaf model. extract pedestrian's bounding boxes from key-points.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """

    def _load_engine(self):
        precision=int(self.config.DETECTOR_TENSORRT_PRECISION)
        TRTbinPath='/repo/applications/facemask/data/tensorrt/openpifpaf_resnet50_{}_{}_d{}.trt'.format(self.w,self.h,precision)
        if not os.path.exists(TRTbinPath):
            os.system('bash /repo/applications/facemask/generate_tensorrt.bash /repo/applications/facemask/configs/config_jetson.json 1')
        with open(TRTbinPath, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:    
            return runtime.deserialize_cuda_engine(f.read())


    def __init__(self, config):
        self.config = config
        self.fps = None
        [self.w, self.h] = self.config.DETECTOR_INPUT_SIZE
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.model_input_size = (self.w, self.h)
        self.device = None  # enter your Gpu id here
        self.cuda_context = None 
        self._init_cuda_stuff()
    def _init_cuda_stuff(self):
        cuda.init()
        self.engine = self._load_engine()
        self.device = cuda.Device(0)  # enter your Gpu id here
        self.cuda_context = self.device.make_context()
        self.engine_context = self.engine.create_execution_context()
        bindings, host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream = allocate_buffers(self.engine)
        self.bindings = bindings
        self.host_inputs = host_inputs
        self.host_outputs = host_outputs
        self.cuda_inputs = cuda_inputs
        self.cuda_outputs = cuda_outputs
        self.stream = stream 

    def __del__(self):
        """ Free CUDA memory. """

        self.cuda_context.pop()
        del self.cuda_context
        del self.engine_context
        del self.engine


    def inference(self, resized_rgb_image):
        """
        This method will perform inference and return the detected bounding boxes
        Args:
            resized_rgb_image: uint8 numpy array with shape (img_height, img_width, channels)

        Returns:
            result: a dictionary contains of [{"id": 0, "bbox": [x1, y1, x2, y2], "score":s%}, {...}, {...}, ...]

        """
        image = resized_rgb_image
        image = cv2.resize(image, self.model_input_size)
        pil_im = PIL.Image.fromarray(image)
        preprocess = None

        data = openpifpaf.datasets.PilImageList([pil_im], preprocess=preprocess)
        loader = torch.utils.data.DataLoader(
            data, batch_size=1, shuffle=False, pin_memory=True,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)
        
        for images_batch, _, __ in loader:
            np_img = images_batch.numpy()

                
        bindings = self.bindings
        host_inputs = self.host_inputs
        host_outputs = self.host_outputs
        cuda_inputs = self.cuda_inputs
        cuda_outputs = self.cuda_outputs
        stream = self.stream
        
        host_inputs[0] = np.ravel(np.zeros_like(np_img))

        self.cuda_context.push()
        t_begin = time.perf_counter()

        np.copyto(host_inputs[0], np.ravel(np_img))       
        cuda.memcpy_htod_async( 
            cuda_inputs[0], host_inputs[0], stream)       

        self.engine_context.execute_async(
            batch_size=1,       
            bindings=bindings,  
            stream_handle=stream.handle)
        cif=[None] * 1          
        caf=[None] * 1          
        cif_names=['cif']       
        caf_names=['caf']       
        for i in range(1, self.engine.num_bindings):      
            cuda.memcpy_dtoh_async(    
                host_outputs[i - 1], cuda_outputs[i - 1], stream)     

        stream.synchronize()    

        for i in range(1, self.engine.num_bindings):      
            shape = self.engine.get_binding_shape(i)      
            name = self.engine.get_binding_name(i)        
            total_shape = np.prod(shape)
            output = host_outputs[i - 1][0: total_shape]  
            output = np.reshape(output, tuple(shape))     
            if name in cif_names:      
                index_n = cif_names.index(name)           
                tmp = torch.from_numpy(output[0])         
                cif = tmp.cpu().numpy()
            elif name in caf_names:    
                index_n = caf_names.index(name)           
                tmp = torch.from_numpy(output[0])         
                caf = tmp.cpu().numpy()


        heads = [cif, caf]    
        self.cuda_context.pop() 
        
        inference_time = time.perf_counter() - t_begin
       
        fields = heads 
        
        decoder_time=time.perf_counter()
        decoder = CifCafDecoder()
        predictions = decoder.decode(fields)
        decoder_time = time.perf_counter() - t_begin
        self.fps = convert_infr_time_to_fps(inference_time+decoder_time)
        #print(f'inference time is {inference_time} and decoder time is :{decoder_time}') 
        result = []

     
        for i, pred_object in enumerate(predictions):
            pred = pred_object.data
            pred_visible = pred[pred[:, 2] > .2]
            xs = pred_visible[:, 0]
            ys = pred_visible[:, 1]
            
            if len(xs) == 0 or len(ys) == 0:
                continue

            x, y, w, h = pred_object.bbox()

            x_min = int(x)
            x_max = int(x + w)
            y_min = int(y)
            y_max = int(y + h)
            xmin = int(max(x_min - .15 * w, 0))
            xmax = int(min(x_max + .15 * w, self.w))
            ymin = int(max(y_min - .2 * h, 0))
            ymax = int(min(y_max + .05 * h, self.h))
            bbox_dict={}
            
            # extract face bounding boxi
            if np.all(pred[[0, 1, 2, 5, 6], -1] > 0.15):
                x_min_face = int(pred[6, 0])
                x_max_face = int(pred[5, 0])
                y_max_face = int((pred[5, 1] + pred[6, 1]) / 2)
                y_eyes = int((pred[1, 1] + pred[2, 1]) / 2)
                y_min_face = 2 * y_eyes - y_max_face
                if (y_max_face - y_min_face > 0) and (x_max_face - x_min_face > 0):
                    h_crop = y_max_face - y_min_face
                    x_min_face = int(max(0, x_min_face - 0.05 * h_crop))
                    y_min_face = int(max(0, y_min_face - 0.05 * h_crop))
                    x_max_face = int(min(self.w, x_min_face + 1.1 * h_crop))
                    y_max_face = int(min(self.h, y_min_face + 1.1 * h_crop))
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
