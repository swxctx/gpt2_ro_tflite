import tensorflow as tf

# 加载刚刚转换成的TensorFlow模型
converter = tf.lite.TFLiteConverter.from_saved_model('./model/tf_model')

# 启用TF Select Ops
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,  # 启用TFLite内置Ops
  tf.lite.OpsSet.SELECT_TF_OPS  # 启用TF Select Ops以支持更多的TensorFlow操作
]

# 进行转换
tflite_model = converter.convert()

# 保存.tflite模型
with open('./model/gpt2.tflite', 'wb') as f:
    f.write(tflite_model)
