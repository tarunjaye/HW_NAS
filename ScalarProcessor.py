import math

class ScalarProcessor:
    def __init__(self, data_forwading, pipeline_overhead, vector_width, multiplier_size):
        self.no_of_lanes = data_forwading
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

        cycles_to_perform_one_mac =  7   # 2 for load, 5 for MAC

        cc_to_clear_acc = 0

        # Cycles to output one pixel
        cycles_to_output_one_pixel = kernel_size[0] * kernel_size[1] * cycles_to_perform_one_mac
        cycles_to_put_it_into_l1cache = 0 # Assuming constant 2 cc
        
        # Total cycles to output one pixel to L1 cache
        total_cycles_to_output_one_pixel_to_l1_cache = (kernel_size[0] * kernel_size[0] * cycles_to_perform_one_mac) + cycles_to_put_it_into_l1cache + cc_to_clear_acc

        

        total_cc_to_finish_convolution_for_output_size = total_cycles_to_output_one_pixel_to_l1_cache * output_pixels * input_shape[0]

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

        cycles_to_perform_one_mac =  8   # 2 for load, 5 for MAC

        # Total MAC operations required for the entire GEMM layer
        total_mac_operations = macs_per_output_element * output_size
        print(f"Total MAC operations for GEMM: {total_mac_operations}")

        # Calculate MAC operations per cycle count
        mac_operations_per_cc = self.vector_width // self.multiplier_size

        # Total cycles to finish GEMM operation
        total_cc_to_finish_gemm = total_mac_operations  * cycles_to_perform_one_mac + self.pipeline_overhead

        print("Total cycles to finish GEMM:", total_cc_to_finish_gemm)

        return total_cc_to_finish_gemm

    
    
    def calculate_maxpooling_cc(self, input_shape, kernel_shape, output_shape):
        # Step 1: Calculate the number of elements inside the pooling window
        elements_inside_pooling_window = kernel_shape[0] * kernel_shape[1]
        
        # Step 2: Calculate the number of pooling windows (output shape height * width)
        no_of_pooling_windows = output_shape[1] * output_shape[2]

        print(f"Number of  pooling windows : {no_of_pooling_windows}")

        constant_overhead = 150

        # Step 3: Calculate the total cycles per pooling window
        total_cycles_per_window = (
            elements_inside_pooling_window +  # Loads
            (elements_inside_pooling_window - 1) +  # Comparisons
            (elements_inside_pooling_window - 1) + # Jumps
            2
        )
        print(f"Total cycles per pooling window: {total_cycles_per_window}")
        
        # Step 4: Calculate the total cycles for all pooling windows
        total_cc = total_cycles_per_window * no_of_pooling_windows * input_shape[0]
        print(f"Total cycles for all pooling windows: {total_cc}")
        
        return total_cc
    
    def calculate_avgpooling_cc(self, input_shape, kernel_shape, output_shape):

        division_overhead = 10
        constant_overhead = 150
        # Step 1: Calculate the number of elements inside the pooling window
        elements_inside_pooling_window = kernel_shape[0] * kernel_shape[1]
        
        # Step 2: Calculate the number of pooling windows (output shape height * width)
        no_of_pooling_windows = output_shape[1] * output_shape[2]

        print(f"Number of  pooling windows : {no_of_pooling_windows}")
        
        # Step 3: Calculate the total cycles per pooling window
        total_cycles_per_window = (
            elements_inside_pooling_window +  # Loads
            (elements_inside_pooling_window - 1) +  # adds
            division_overhead  # Jumps
        )
        print(f"Total cycles per pooling window: {total_cycles_per_window}")
        
        # Step 4: Calculate the total cycles for all pooling windows
        total_cc = total_cycles_per_window * no_of_pooling_windows * input_shape[0]
        print(f"Total cycles for all pooling windows: {total_cc}")
        
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

         # Step 2: Calculate the total cycles for the ReLU operation
        total_cycles = total_pixels * 4 * no_of_channels  # 3 operations per element: Load, CMP, Conditional Move
        print(f"Total cycles for ReLU operation: {total_cycles}")
        return total_cycles
    
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
        

        # Step 2: Calculate the total cycles for the ReLU operation
        total_cycles = total_pixels * 6 * no_of_channels  # 3 operations per element: Load, CMP, Conditional Move
        print(f"Total cycles for ReLU operation: {total_cycles}")

        return total_cycles

