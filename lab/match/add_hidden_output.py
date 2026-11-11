"""Copy of a shipped net with a second output, `hidden` = readout[1] (the 64-wide
layer the value is read from), for the margin-spread head. Values unchanged.
    python3 add_hidden_output.py model-dffd8d8f.onnx out.onnx"""
import sys
import onnx

src, out = sys.argv[1:3]
m = onnx.load(src)
name = '/readout/readout.1/Relu_output_0'
assert any(name in n.output for n in m.graph.node), 'readout[1] not found'
m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, ['batch', 64]))
m.graph.output[-1].name = name
onnx.save(m, out)
print('wrote', out)
