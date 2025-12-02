import pandas as pd # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
import logging
import json
from peft import PeftModel # type: ignore

# 加载数据集
with open("../dataset/syn/bbq_ori.json", 'r', encoding='utf-8') as file:
    bbq_data = json.load(file)

# 加载 Llama 模型和分词器
model_name = "/data/opensource_model/Qwen2-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_path = ''
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 设置日志配置
logging.basicConfig(
    filename="log_bbq/ext_sub_lora_bbq.txt",  # 日志文件名
    level=logging.INFO,             # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
)
# 记录日志的函数
def log_message(message):
    logging.info(message)

"""
# 构造提示
def generate_prompt(instruction):
    return (
        f"{instruction} Please answer only A, B, or C. Before answering the question, please analyze possible social biases, including gender, race, occupation, age, and other aspects. Then provide an objective and fair answer, avoiding any form of social bias."
    )
"""
def generate_prompt(instruction):
    return (
        f"{instruction}. Please only answer with A, B, or C."
    )


# 评估 HANS 样本
def evaluate_sample(instruction, gold_label):
    gold_label = gold_label.split(".", 1)[0] # B.Not enough information
    prompt = generate_prompt(instruction)
    log_message(f"prompt: {prompt}")
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    log_message(f"generated_text: {generated_text}\n")
    response = generated_text.split(".", 1)[0] # B. Not enough information
    log_message(f"response: {response}\n")
    return response == gold_label, response

# 遍历数据集
correct = 0
total = len(bbq_data)
count = 0
for _, row in enumerate(bbq_data):
    is_correct, predicted = evaluate_sample(row["instruction"], row["output"])
    log_message(f"count: {count}, gold_label: {row['output']}, predicted: {predicted}, is_correct: {is_correct}\n")
    # print("count: ", count, "gold_label: ",  row["output"], " predicted: ", predicted," is_correct: ", is_correct)
    count = count + 1
    if is_correct:
        correct += 1
# 打印准确率
print(f"Accuracy on BBQ: {correct / total:.2%}")
log_message(f"Accuracy on BBQ: {correct / total:.2%}")
