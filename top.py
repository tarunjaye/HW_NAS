from ONNXModel import ONNXModel
from LayerProcessor import LayerProcessor

# Usage example
model_path = "/content/age_googlenet.onnx"
output_file = "processed_layers.txt"

onnx_model = ONNXModel(model_path)
layers_info = onnx_model.extract_layers_info()

for layer_info in layers_info:
    if layer_info["Type"] == "Conv":
        LayerProcessor.process_layer(layer_info["Name"], layer_info["Type"], layer_info.get("Attributes", {}), output_file)
