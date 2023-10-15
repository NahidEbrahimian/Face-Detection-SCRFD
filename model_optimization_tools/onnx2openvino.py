from openvino.runtime import Core
import openvino.runtime as ov 
from openvino.tools import mo 


onnx_path = "./scrfd_500m_bnkps.onnx"
ir_path = "./scrfd_500m_bnkps.xml"


ov_model = mo.convert_model(onnx_path, model_name=ir_path, 
                            framework='onnx', compress_to_fp16=True) #, compress_to_fp16=half)  # export


# Assign dynamic shapes
shapes = {}
for input_layer in ov_model.inputs:
    shapes[input_layer] = input_layer.partial_shape
    shapes[input_layer][2] = -1
    shapes[input_layer][3] = -1
ov_model.reshape(shapes)

core = ov.Core()
ov.serialize(ov_model, ir_path)  # save

# Print model input layer info
dynamic_model = core.read_model(ir_path)
for input_layer in dynamic_model.inputs:
    print(input_layer.names, input_layer.partial_shape)
    