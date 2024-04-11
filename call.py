from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = './model/gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 准备问题和上下文
question = "Who are you?"

# 编码并生成答案
input_ids = tokenizer.encode(question, return_tensors='pt')
output = model.generate(input_ids, max_length=100, temperature=1.0, pad_token_id=tokenizer.eos_token_id)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# 打印生成的文本
print(generated_text)