import onnx
from onnx_tf.backend import prepare

# 加载ONNX模型
onnx_model = onnx.load("./model/gpt2.onnx")

# 转换为TensorFlow模型
tf_rep = prepare(onnx_model)

# 保存TensorFlow模型
tf_rep.export_graph('./model/tf_model')
