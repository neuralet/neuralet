#!/bin/bash
config="$1"
run_on_jetson="$2"
width=$(cat $config  | jq '.detector.input_size[0]')
height=$(cat $config  | jq '.detector.input_size[1]')

workdir="/repo/applications/facemask/"
onnx_dir="${work_dir}data/onnx"
mkdir -p $onnx_dir
onnx_name_openpifpaf="${work_dir}data/onnx/openpifpaf_resnet50_${width}_${height}.onnx"
onnx_name_face_mask="${work_dir}data/onnx/ofm_face_mask.onnx"

onnx_openpifpaf_download_url="https://media.githubusercontent.com/media/neuralet/neuralet-models/c874b2bcee0521d770d3480ed5fef25643160abd/ONNX/openpifpaf_12a4/openpifpaf_resnet50_${width}_${height}.onnx"
if [[ ! $run_on_jetson ]] && [[ ! -f $onnx_name_openpifpaf ]]; then
    echo "############## exporting ONNX from OpenPifPaf ##################"
    python3 -m openpifpaf.export_onnx --outfile $onnx_name_openpifpaf  --checkpoint resnet50  --input-width $width --input-height $height
   
elif [[ $run_on_jetson ]] && [[ ! -f $onnx_name_openpifpaf  ]]; then
    wget $onnx_openpifpaf_download_url -O $onnx_name_openpifpaf
fi

if [ ! -f $onnx_name_face_mask ]; then
    wget https://media.githubusercontent.com/media/neuralet/neuralet-models/c874b2bcee0521d770d3480ed5fef25643160abd/ONNX/OFMClassifier/OFMClassifier.onnx -O $onnx_name_face_mask
fi
tensorrt_dir="${work_dir}data/tensorrt/"
mkdir -p $tensorrt_dir

precision_detector=$(cat $config  | jq '.detector.tensorrt_precision')
tensorrt_name_openpifpaf="${tensorrt_dir}openpifpaf_resnet50_${width}_${height}_d${precision_detector}.trt"

if [ ! -f $tensorrt_name_openpifpaf ]; then
    echo "############## Generating TensorRT Engine for openpifpaf ######################"
    onnx2trt $onnx_name_openpifpaf -o $tensorrt_name_openpifpaf -d $precision_detector -b 1
fi
precision_classifier=$(cat $config  | jq '.classifier.tensorrt_precision')
tensorrt_name_face_mask="${tensorrt_dir}ofm_face_mask_d${precision_classifier}.trt"

if [ ! -f $tensorrt_name_face_mask ]; then
    echo "############## Generating TensorRT Engine for face_mask classifier ##############"
    onnx2trt $onnx_name_face_mask -o $tensorrt_name_face_mask -d $precision_classifier -b 1
fi

