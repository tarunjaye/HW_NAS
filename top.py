import onnx
from VectorProcessor import VectorProcessor
from ScalarProcessor import ScalarProcessor


class LayerInfo:
    def __init__(self, name, layer_type, input_shape, output_shape, kernel_shape=None, padding=None, strides=None):
        self.name = name
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.strides = strides

    def __str__(self):
        info_str = f"  Type: {self.layer_type}\n"
        info_str += f"  Input Shape: {self.input_shape}\n"
        info_str += f"  Output Shape: {self.output_shape}\n"
        if self.kernel_shape:
            info_str += f"  Kernel Size: {self.kernel_shape}\n"
        if self.padding:
            info_str += f"  Padding: {self.padding}\n"
        if self.strides:
            info_str += f"  Stride: {self.strides}\n"
        return info_str
    

class ONNXModel:
    def __init__(self, model_path, vector_processor, scalar_processor):
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.input_shape = self._get_model_input_shape()
        self.layers_info = []
        self.vector_processor = vector_processor  # Pass the VectorProcessor instance
        self.scalar_processor = scalar_processor

    def _get_model_input_shape(self):
        input_tensor = self.graph.input[0]
        return [dim.dim_value if dim.dim_value is not None else 1 for dim in input_tensor.type.tensor_type.shape.dim]

    def _calculate_conv_output_shape(self, input_shape, kernel_shape, pads, strides):
        if len(pads) == 2:
            pads = [pads[0], pads[0], pads[1], pads[1]]

        output_shape = [
            (input_shape[0] + 2 * pads[0] - kernel_shape[0]) // strides[0] + 1,
            (input_shape[1] + 2 * pads[2] - kernel_shape[1]) // strides[1] + 1
        ]
        return output_shape

    def _calculate_pooling_output_shape(self, input_shape, kernel_shape, pads, strides):
        return self._calculate_conv_output_shape(input_shape, kernel_shape, pads, strides)

    def extract_layers_info(self):
        input_shape = self.input_shape[1:]  # Remove batch size for calculation
        for node in self.graph.node:
            layer_type = node.op_type
            output_shape = None
            attributes = {attr.name: list(attr.ints) for attr in node.attribute}

            if layer_type == "Conv":
                kernel_shape = attributes.get("kernel_shape", [1, 1])
                pads = attributes.get("pads", [0, 0, 0, 0])
                strides = attributes.get("strides", [1, 1])
                padding = [pads[0], pads[2]] if len(pads) == 4 else pads

                output_shape = self._calculate_conv_output_shape(input_shape[1:], kernel_shape, padding, strides)
                output_shape = [input_shape[0]] + output_shape  # Add batch size back

                # Calculate cycle count using VectorProcessor
                cc = self.vector_processor.calculate_convolution_cc(input_shape, kernel_shape, output_shape)
                cc_scalar = self.scalar_processor.calculate_convolution_cc(input_shape, kernel_shape, output_shape)

                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape, kernel_shape, padding, strides)
                print(f"Layer {node.name} (Conv) - CC: {cc}")  # Display the cycle count
                print(f"Layer {node.name} (Conv_scalar) - CC: {cc_scalar}")  # Display the cycle count


            elif layer_type == "MaxPool":
                kernel_shape = attributes.get("kernel_shape", [1, 1])
                pads = attributes.get("pads", [0, 0, 0, 0])
                strides = attributes.get("strides", [1, 1])
                padding = [pads[0], pads[2]] if len(pads) == 4 else pads

                output_shape = self._calculate_pooling_output_shape(input_shape[1:], kernel_shape, padding, strides)
                output_shape = [input_shape[0]] + output_shape  # Add batch size back

                print(f"max pool output shape : {output_shape}")
                
                # Calculate cycle count for pooling using VectorProcessor
                cc = self.vector_processor.calculate_maxpooling_cc(input_shape, kernel_shape, output_shape)
                cc_scalar = self.scalar_processor.calculate_maxpooling_cc(input_shape, kernel_shape, output_shape)

                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape, kernel_shape, padding, strides)
                print(f"Layer {node.name} (Pooling) - CC: {cc}")  # Display the cycle count
                print(f"Layer {node.name} (Pooling_scalar) - CC: {cc_scalar}")  # Display the cycle count

            elif layer_type == "AveragePool":
                kernel_shape = attributes.get("kernel_shape", [1, 1])
                pads = attributes.get("pads", [0, 0, 0, 0])
                strides = attributes.get("strides", [1, 1])
                padding = [pads[0], pads[2]] if len(pads) == 4 else pads

                # Calculate output shape for Average Pooling
                output_shape = self._calculate_pooling_output_shape(input_shape[1:], kernel_shape, padding, strides)
                output_shape = [input_shape[0]] + output_shape  # Add batch size back

                print(f"Average Pool output shape: {output_shape}")
    
                # Calculate cycle count for Average Pooling
                # Assuming you have a similar method for Average Pooling in VectorProcessor
                cc = self.vector_processor.calculate_avgpooling_cc(input_shape, kernel_shape, output_shape)

                # Store or process layer info for Average Pool
                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape, kernel_shape, padding, strides)
                print(f"Layer {node.name} (Average Pooling) - CC: {cc}")  # Display the cycle count

            elif layer_type == "Gemm":
                print("gemm input shape", input_shape)
                output_size = attributes.get("output_size", [input_shape[0]])[0]
                output_shape = [output_size]

                print("gemm output shape", output_shape)

                # Calculate cycle count for fully connected (Gemm) using VectorProcessor
                cc = self.vector_processor.calculate_gemm_cc(input_shape, output_shape)
                cc_scalar = self.scalar_processor.calculate_gemm_cc(input_shape, output_shape)

                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape)
                print(f"Layer {node.name} (Gemm) - CC: {cc}")  # Display the cycle count
                print(f"Layer {node.name} (Gemm_scalar) - CC: {cc_scalar}")  # Display the cycle count

            elif layer_type == "Relu":
                output_shape = input_shape  # No shape change for ReLU


                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape)
                print(f"Layer Info (before CC calculation):\n{layer_info}")

                # Calculate cycle count for ReLU using VectorProcessor
                cc = self.vector_processor.calculate_relu_cc(input_shape)
                cc_scalar = self.scalar_processor.calculate_relu_cc(input_shape)

                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape)
                print(f"Layer {node.name} (ReLU) - CC: {cc}")  # Display the cycle count
                print(f"Layer {node.name} (ReLU_scalar) - CC: {cc_scalar}")  # Display the cycle count

            elif layer_type == "LeakyRelu":
                output_shape = input_shape  # No shape change for LeakyReLU

                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape)
                print(f"Layer Info (before CC calculation):\n{layer_info}")

                # Calculate cycle count for Leaky ReLU using VectorProcessor
                # Assuming you have or will implement a similar method for Leaky ReLU in the VectorProcessor
                cc = self.vector_processor.calculate_leakyrelu_cc(input_shape)

                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape)
                print(f"Layer {node.name} (Leaky ReLU) - CC: {cc}")  # Display the cycle count


            else:
                output_shape = input_shape  # Default to no shape change
                layer_info = LayerInfo(node.name, layer_type, input_shape, output_shape)

            self.layers_info.append(layer_info)
            input_shape = output_shape  # Set next layer's input shape to current layer's output shape

    def print_layers_info(self):
        for idx, layer in enumerate(self.layers_info, start=1):
            print(f"Layer {idx}:")
            print(layer)


if __name__ == "__main__":
    model_path = "/Users/tarunjaye/Downloads/bvlcalexnet-9.onnx"  # Replace with your ONNX model path
    vector_processor = VectorProcessor(no_of_lanes=4, pipeline_overhead=10,vector_width=256,multiplier_size=32)  # Example values
    scalar_processor = ScalarProcessor(no_of_lanes=4, pipeline_overhead=10,vector_width=256,multiplier_size=32)  # Example values
    onnx_model = ONNXModel(model_path, vector_processor, scalar_processor)
    onnx_model.extract_layers_info()
    onnx_model.print_layers_info()
