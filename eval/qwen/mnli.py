import pandas as pd # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
import logging
from peft import PeftModel # type: ignore
import json
import random
random.seed(42)

with open("../dataset/syn/anli.json", 'r', encoding='utf-8') as file:
    mnli_data = json.load(file)

# random_sample = random.sample(mnli_data, 2000)

model_name = "/data/opensource_model/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# lora权重路径
lora_path = ''

model = PeftModel.from_pretrained(model, model_id=lora_path)

logging.basicConfig(
    filename="log_prompt_en/debias_lap_neg_anli.txt", 
    level=logging.INFO,          
    # format="%(asctime)s - %(levelname)s - %(message)s", 
)

def log_message(message):
    logging.info(message)

"""
def generate_prompt(instruction):
    return (
        f"{instruction}"
    )
"""

def generate_prompt(instruction):
    return (
        f"{instruction} Note that there are bias between the premise and hypothesis, such as overlap bias and negation bias. If there is an overlap bias, do not judge the relationship between premises and hypothesis based on the degree of word overlap between them. For example, the premise 'The artist said that the athletes saw the president .' and the hypothesis 'The athletes saw the president .', their relationship is not 'entailment'. If there is a negative bias, please do not judge the relationship between them based on negative words, such as 'not', 'nobody', 'no', 'never', 'nothing', etc. For example, the premise 'Supposedly the tourist contacted the author.' and the hypothesis 'The tourist contacted the author.', their relationship is also not 'entailment'."
    )

def evaluate_sample(instruction, gold_label):
    if gold_label != "entailment":
        gold_label = "non-entailment"
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
    # log_message(f"generated_text: {generated_text}")
    response = generated_text.replace(prompt, "")
    log_message(f"response: {response}")
    if "entailment" == response.lower():
        predicted_label = "entailment"
    else:
        predicted_label = "non-entailment"
    return predicted_label == gold_label, predicted_label

correct = 0
total = len(mnli_data)
count = 0
for row in mnli_data:
    is_correct, predicted = evaluate_sample(row["instruction"], row["output"])
    log_message(f"count: {count}, gold_label: {row['output']}, predicted: {predicted}, is_correct: {is_correct}")
    if is_correct:
        correct += 1
    count = count + 1
print(f"LoRA Accuracy: {correct / total:.2%}")
log_message(f"LoRA Accuracy: {correct / total:.2%}")
