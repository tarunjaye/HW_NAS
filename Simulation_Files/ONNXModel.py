import onnx

class ONNXModel:
    def __init__(self, model_path):
        self.model = onnx.load(model_path)
        self.graph = self.model.graph

    def extract_layers_info(self):
        layers_info = []
        activation_layer_types = {"Relu", "Sigmoid", "Tanh", "LeakyRelu", "PRelu", "Softmax", "Elu"}

        for node in self.graph.node:
            layer_info = {
                "Name": node.name,
                "Type": node.op_type,
                "Attributes": {}
            }
            if node.op_type == "Conv":
                for attr in node.attribute:
                    if attr.name == "kernel_shape":
                        layer_info["Attributes"]["Kernel Size"] = [int(val) for val in attr.ints]
                    elif attr.name == "pads":
                        layer_info["Attributes"]["Padding"] = [int(val) for val in attr.ints]
                    elif attr.name == "strides":
                        layer_info["Attributes"]["Stride"] = [int(val) for val in attr.ints]
            elif node.op_type in activation_layer_types:
                layer_info["Attributes"]["Activation"] = node.op_type
                
            layers_info.append(layer_info)
        
                # Write the information to a file
        with open("layers_info.txt", "w") as f:
            for layer_info in layers_info:
              f.write("Layer: {}\n".format(layer_info["Name"]))
              f.write("  Type: {}\n".format(layer_info["Type"]))
              for attr_name, attr_value in layer_info["Attributes"].items():
                f.write("  {}: '{}'\n".format(attr_name, attr_value))
              f.write("\n")
        
        return layers_info



