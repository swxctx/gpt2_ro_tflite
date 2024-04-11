from transformers import GPT2Tokenizer

# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained('./model/gpt2')

# token ID列表
token_ids = [318,345,1701,198,198,198]

# 将token ID列表转换为文本并拼接
text = ""
for token_id in token_ids:
    word = tokenizer.decode([token_id])
    text += word

print("Answer: " + text)
