class LayerProcessor:
    # Load inputs from file
    input_size = 6  # Sample value, replace with actual value from the input file
    pipeline_stage_overhead = 5  # Sample value, replace with actual value from the input file
    cache_hit = 1  # Sample value, replace with actual value from the input file
    data_forwarding = 1  # Sample value, replace with actual value from the input file

    @staticmethod
    def calculate_cc_for_filtering(kernel_size, input_size, padding, stride, pipeline_stage_overhead, cache_hit,
                                   data_forwarding):
        # Calculate output_pixels based on the given formula
        print("Kernel Size:", kernel_size)
        print("Input Size:", input_size)
        print("Padding:", padding)
        print("Stride:", stride)
        print("Pipeline Stage Overhead:", pipeline_stage_overhead)
        print("Cache Hit:", cache_hit)
        print("Data Forwarding:", data_forwarding)

        output_size = (input_size - kernel_size[0] + 2 * padding[0]) // stride[0] + 1

        output_pixels = output_size * output_size

        # Cycles to perform one MAC (Multiply-Accumulate operation)
        cycles_to_perform_one_mac = 2 + 1  # 2 for load, 1 for MAC

        # Cycles to output one pixel
        cycles_to_output_one_pixel = kernel_size[0] * kernel_size[0] * cycles_to_perform_one_mac
        cycles_to_put_it_into_l1cache = 2  # Assuming constant 2 cc

        # Total cycles to output one pixel to L1 cache
        total_cycles_to_output_one_pixel_to_l1_cache = (
                    kernel_size[0] * kernel_size[0] * cycles_to_perform_one_mac) + cycles_to_put_it_into_l1cache

        # Cycles to filtering
        cycles_to_filtering = total_cycles_to_output_one_pixel_to_l1_cache * output_pixels + pipeline_stage_overhead

        print("Cycles to output the filtering:", cycles_to_filtering)

        return cycles_to_filtering

    @staticmethod
    def process_layer(name, layer_type, attributes, output_file):
        # Prepare lines to write to the output file
        output_lines = ["Layer: {}\n".format(name), "  Type: {}\n".format(layer_type)]

        # Check if attributes are present
        if attributes:
            output_lines.append("Attributes:\n")
            # Check if 'Kernel Size' attribute exists
            if "Kernel Size" in attributes:
                print("Kernel Size attribute detected:", attributes["Kernel Size"])
                output_lines.append("  Kernel Size: {}\n".format(attributes["Kernel Size"]))
            # Check if 'Stride' attribute exists
            if "Stride" in attributes:
                print("Stride attribute detected:", attributes["Stride"])
                output_lines.append("  Stride: {}\n".format(attributes["Stride"]))
            # Check if 'Padding' attribute exists
            if "Padding" in attributes:
                print("Padding attribute detected:", attributes["Padding"])
                output_lines.append("  Padding: {}\n".format(attributes["Padding"]))
            # Call calculate_cc_for_filtering function if it's a convolutional layer
            if layer_type == "Conv":
                print("Layer is Convolutional")
                print("Attributes:", attributes)
                # Ensure all attributes are present before calling the function
                if "Kernel Size" in attributes and "Stride" in attributes and "Padding" in attributes:
                    print("Calling calculate_cc_for_filtering")
                    cc_result = LayerProcessor.calculate_cc_for_filtering(attributes["Kernel Size"],
                                                                          LayerProcessor.input_size,
                                                                          attributes["Padding"],
                                                                          attributes["Stride"],
                                                                          LayerProcessor.pipeline_stage_overhead,
                                                                          LayerProcessor.cache_hit,
                                                                          LayerProcessor.data_forwarding)
                    output_lines.append("  CC Result: {}\n".format(cc_result))
        else:
            output_lines.append("No attributes found for this layer.\n")

        output_lines.append("-----------------------------------\n")

        # Write lines to the output file
        with open(output_file, "a") as f:
            f.writelines(output_lines)
