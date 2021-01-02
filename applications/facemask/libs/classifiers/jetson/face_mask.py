import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import sys 
import time


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

class Classifier():
    
    """
    Perform image classification with the given model. The model is a tensorrt file
    which if the classifier can not find it at the path it will generate it
    from provided ONNX file.
    :param config: Is a ConfigEngine instance which provides necessary parameters.
    """
    
    def _load_engine(self):
        precision=int(self.config.CLASSIFIER_TENSORRT_PRECISION)
        TRTbinPath='/repo/applications/facemask/data/tensorrt/ofm_face_mask_d{}.trt'.format(precision)
        with open(TRTbinPath, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, config):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = ''
        self.fps = None
        self.config = config
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.trt_bin_path = self.config.CLASSIFIER_MODEL_PATH

        self.model_input_size = self.config.CLASSIFIER_INPUT_SIZE 
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
        """ Free CUDA memories. """

        self.cuda_context.pop()
        del self.cuda_context
        del self.engine_context
        del self.engine


    def inference(self, resized_rgb_images):
        """
        Inference function sets input tensor to input image and gets the output.
        The interpreter instance provides corresponding class id output which is used for creating result
        Args:
            resized_rgb_images: Array of images with shape (no_images, img_height, img_width, channels)
        Returns:
            result: List of class id for each input image. ex: [0, 0, 1, 1, 0]
            scores: The classification confidence for each class. ex: [.99, .75, .80, 1.0]
        """
        bindings = self.bindings
        host_inputs = self.host_inputs
        host_outputs = self.host_outputs
        cuda_inputs = self.cuda_inputs
        cuda_outputs = self.cuda_outputs
        stream = self.stream 
        t_begin = time.perf_counter()
        result = []
        scores = []
        for img in resized_rgb_images: 
            img = np.expand_dims(img, axis=0)
            img = img.astype(np.float32)

            host_inputs[0] = np.ravel(np.zeros_like(img)) #np.ravel(np.zeros_like(np_img))
        
            self.cuda_context.push() 
        
            np.copyto(host_inputs[0], img.ravel())
            cuda.memcpy_htod_async(
                cuda_inputs[0], host_inputs[0], stream)
               
            self.engine_context.execute_async(
                batch_size=1,
                bindings=bindings,
                stream_handle=stream.handle)
             
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            stream.synchronize()
            output_dict = host_outputs[0]
            pred = list(np.argmax(host_outputs, axis=1))
            
            # TODO: optimized without for
            for i, itm in enumerate(host_outputs):
                scores.append(itm[pred[i]]) 
            
            result.append(pred[0])
            self.cuda_context.pop()
        if len(resized_rgb_images) > 0:
            inference_time = float(time.perf_counter() - t_begin) / len(resized_rgb_images)
        return result, scores
