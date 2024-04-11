from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 对文本进行编码，获取input IDs
input_ids = tokenizer.encode("Who are you?", return_tensors='pt')

# 得到ID数组后将ID数组复制放到客户端输入框
print(input_ids)
