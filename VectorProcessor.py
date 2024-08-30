import math

class VectorProcessor:
    def __init__(self, no_of_lanes, pipeline_overhead, vector_width, multiplier_size):
        self.no_of_lanes = no_of_lanes
        self.pipeline_overhead = pipeline_overhead
        self.vector_width = vector_width
        self.multiplier_size = multiplier_size

    def calculate_convolution_cc(self, input_shape, kernel_shape, output_shape):
        # Extracting the relevant sizes
        output_size = output_shape[1]  # assuming square output shape [channels, height, width]
        kernel_size = kernel_shape  # assuming square kernel [height, width]

        # Calculating output pixels (assuming square output)
        output_pixels = output_size * output_size

        # Calculating MAC operations per cycle count
        mac_operations_per_cc = self.vector_width / self.multiplier_size

        # Cycles to perform MAC operations for one output element
        cc_to_mac_one_output_element = math.ceil(kernel_size[0] * kernel_size[1] / mac_operations_per_cc)

        # Cycles to store one output element
        cc_to_store_one_output_element = 1

        # Cycles to clear accumulator
        cc_to_clear_acc = 1

        # Total cycles to finish convolution for the entire output size,
        # multiplied by 5 times the output pixels
        total_cc_to_finish_convolution_for_output_size = ((
            (cc_to_mac_one_output_element + cc_to_store_one_output_element + cc_to_clear_acc) * 5 * output_pixels 
        ) + self.pipeline_overhead )* input_shape[0]

        print("Vector_processor: Cycles to output the filtering:", total_cc_to_finish_convolution_for_output_size)

        return total_cc_to_finish_convolution_for_output_size

    

    def calculate_gemm_cc(self, input_shape, output_shape):
        print("Input shape:", input_shape)

        # Flatten the input shape if necessary
        if len(input_shape) > 1:
            # Flatten all dimensions to a single value
            flattened_input_size = math.prod(input_shape)
        else:
            # If input shape is already one-dimensional
            flattened_input_size = input_shape[0]

        print("Flattened input size:", flattened_input_size)

        # Output size is the number of output elements (typically the number of neurons)
        output_size = output_shape[0]

        # MACs required for one output element
        macs_per_output_element = flattened_input_size
        print(f"MACs per output element: {macs_per_output_element}")

        # Total MAC operations required for the entire GEMM layer
        total_mac_operations = macs_per_output_element * output_size
        print(f"Total MAC operations for GEMM: {total_mac_operations}")

        # Calculate MAC operations per cycle count
        mac_operations_per_cc = self.vector_width // self.multiplier_size

        # Total cycles to finish GEMM operation
        total_cc_to_finish_gemm = math.ceil(total_mac_operations / mac_operations_per_cc) * 5 + self.pipeline_overhead

        print("Total cycles to finish GEMM:", total_cc_to_finish_gemm)

        return total_cc_to_finish_gemm

    
    
    def calculate_maxpooling_cc(self, input_shape, kernel_shape, output_shape):
        # Step 1: Calculate the number of elements inside the pooling window
        elements_inside_pooling_window = kernel_shape[0] * kernel_shape[1]
        
        # Step 2: Calculate the number of pooling windows (output shape height * width)
        no_of_pooling_windows = output_shape[1] * output_shape[2]

        print(f"Number of  pooling windows : {no_of_pooling_windows}")
        
        # Step 3: Calculate the number of registers required
        no_of_regs_required = elements_inside_pooling_window
        
        # Step 4: Calculate the number of lanes (or windows that can be processed simultaneously)
        lanes = self.vector_width // self.multiplier_size

        print(f"Number of  lanes : {lanes}")
        
        # Step 5: Calculate the cycles to finish max pooling for the number of lanes
        total_cc_to_finish_max_for_lanes = no_of_regs_required - 1
        
        print(f"total_cc_to_finish_max_for_lanes : {total_cc_to_finish_max_for_lanes}")
        
        
        # Step 6: Calculate the total number of cycles for all pooling windows
        sets_of_lanes = math.ceil(no_of_pooling_windows / lanes)

        print(f"sets of lanes : {sets_of_lanes}")
        
        # Total cycle count to finish max pooling (without pipeline overhead for debugging)
        total_cc = (sets_of_lanes * total_cc_to_finish_max_for_lanes + self.pipeline_overhead) * input_shape[0]
        
        return total_cc
    
    def calculate_avgpooling_cc(self, input_shape, kernel_shape, output_shape):

        division_overhead = 4
        # Step 1: Calculate the number of elements inside the pooling window
        elements_inside_pooling_window = kernel_shape[0] * kernel_shape[1]
        
        # Step 2: Calculate the number of pooling windows (output shape height * width)
        no_of_pooling_windows = output_shape[1] * output_shape[2]

        print(f"Number of  pooling windows : {no_of_pooling_windows}")
        
        # Step 3: Calculate the number of registers required
        no_of_regs_required = elements_inside_pooling_window
        
        # Step 4: Calculate the number of lanes (or windows that can be processed simultaneously)
        lanes = self.vector_width // self.multiplier_size

        print(f"Number of  lanes : {lanes}")
        
        # Step 5: Calculate the cycles to finish max pooling for the number of lanes
        total_cc_to_finish_max_for_lanes = no_of_regs_required - 1 + division_overhead
        
        print(f"total_cc_to_finish_max_for_lanes : {total_cc_to_finish_max_for_lanes}")
        
        
        # Step 6: Calculate the total number of cycles for all pooling windows
        sets_of_lanes = math.ceil(no_of_pooling_windows / lanes)

        print(f"sets of lanes : {sets_of_lanes}")
        
        # Total cycle count to finish max pooling (without pipeline overhead for debugging)
        total_cc = (sets_of_lanes * total_cc_to_finish_max_for_lanes + self.pipeline_overhead) * input_shape[0]
        
        return total_cc

    #def calculate_gemm_cc(self, input_shape, output_shape):
        # Example fully connected (Gemm) cycle count calculation
        #cc = input_shape[0] * output_shape[0]  # dot products
        #cc = cc / self.no_of_lanes + self.pipeline_overhead
       # return cc

    def calculate_relu_cc(self, input_shape):
        if len(input_shape) == 3:
           # If input shape is [channels, height, width]
           total_pixels = input_shape[1] * input_shape[2]
           no_of_channels = input_shape[0]
        elif len(input_shape) == 2:
           # If input shape is [batch_size, features]
           total_pixels = input_shape[1]
        elif len(input_shape) == 1:
           # If input shape is [features] or just a single dimension
           total_pixels = input_shape[0]  # Use the single dimension as the total pixel count
           no_of_channels = 1
           print(f"total pixels: {total_pixels}")
        else:
           raise ValueError(f"Unexpected input shape: {input_shape}")

        relu_operations_per_cc = self.vector_width / self.multiplier_size
        print(f"relu ops persec : {relu_operations_per_cc}")
        total_cc_for_relu = math.ceil( total_pixels / relu_operations_per_cc) * no_of_channels
        return total_cc_for_relu
    
    def calculate_leakyrelu_cc(self, input_shape):
        if len(input_shape) == 3:
           # If input shape is [channels, height, width]
           total_pixels = input_shape[1] * input_shape[2]
           no_of_channels = input_shape[0]
        elif len(input_shape) == 2:
           # If input shape is [batch_size, features]
           total_pixels = input_shape[1]
        elif len(input_shape) == 1:
           # If input shape is [features] or just a single dimension
           total_pixels = input_shape[0]  # Use the single dimension as the total pixel count
           no_of_channels = 1
        else:
           raise ValueError(f"Unexpected input shape: {input_shape}")
        

        multiplication_overhead = 3

        relu_operations_per_cc = self.vector_width / self.multiplier_size + multiplication_overhead + self.pipeline_overhead
        total_cc_for_relu = math.ceil( total_pixels / relu_operations_per_cc)  
        return total_cc_for_relu

