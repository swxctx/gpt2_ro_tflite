import torch
from transformers import GPT2Model

model_name = './model/gpt2'  # 本地模型路径
model = GPT2Model.from_pretrained(model_name)

# 设置模型为评估模式
model.eval()

# 创建一个假的输入对应于模型的输入
dummy_input = torch.randint(50257, (1, 3))  # 假设输入的shape为(1, 3)，50257是GPT-2的词汇表大小

# 导出模型
onnx_model_path = './model/gpt2.onnx'
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11,
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
