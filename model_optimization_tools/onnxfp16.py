import argparse
import copy
import os

import onnx
from onnxconverter_common.float16 import convert_float_to_float16

parser = argparse.ArgumentParser(
        description="Convert onnx32 to onnx16")
parser.add_argument("--input_onnx", type=str, help="input model path")
parser.add_argument("--output_onnx", type=str, help="Output model path")
args = parser.parse_args()    

model_fp32 = onnx.load(args.input_onnx)
model_fp16 = convert_float_to_float16(copy.deepcopy(model_fp32))
onnx.save(model_fp16, os.path.join(args.output_onnx, args.input_onnx.split("/")[-1]))
