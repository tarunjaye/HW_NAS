Here all the files required can be found
1.ONNXModel.py:Loads the onnx file and parse the information about layers into layers_info.txt
2.LayerProcessor.py:extracts necessary information from the layers and passes data to clock cycle estimation function and prints output to processed_layers.txt
3.top.py:ONNXModel and Layerprocessor are called from here
