import argparse
import os

import numpy as np
import onnx
import tensorrt as trt

parser = argparse.ArgumentParser(description='Process some aurguments')
parser.add_argument('--onnx_model_address', default="./models/scrfd_500m_bnkps.onnx", type=str, help='ONNX model address')
parser.add_argument('--trt_model_address', default="./models/scrfd_500m_bnkps.engine", type=str, help='Tensorrt modeladdress')
args = parser.parse_args()

onnx_model_path = args.onnx_model_address 
trt_engine_path = args.trt_model_address

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def GiB(val):
    return val * 1 << 30

def build_engine(onnx_model_path, trt_engine_path, fp16_mode=False):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as config:        
        profile = builder.create_optimization_profile()     
        profile.set_shape("input.1", 
                          (1, 3, 640, 640), # min
                          (1, 3, 960, 960), # opt
                          (1, 3, 1280, 1280)) # max
        config.add_optimization_profile(profile)

        config.max_workspace_size=GiB(1) 

        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16) 
        with open(onnx_model_path, 'rb') as model:
            assert parser.parse(model.read())
            serialized_engine=builder.build_serialized_network(network, config)

        with open(trt_engine_path, 'wb') as f:
            f.write(serialized_engine)

        print('TensorRT file in ' + trt_engine_path)
        print('============ONNX->TensorRT SUCCESS============')


if __name__ == "__main__":  
    # build_engine(onnx_model_path, trt_engine_path) #for fp32
    build_engine(onnx_model_path, trt_engine_path,fp16_mode=True)#for fp16