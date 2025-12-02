import torch # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import json
from rouge_score import rouge_scorer  # type: ignore
from peft import PeftModel # type: ignore
import pandas as pd

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# 加载数据集
with open("../dataset/syn/news_unbias.json", 'r', encoding='utf-8') as file:
    news_data = json.load(file)

# 加载 Llama 模型和分词器
model_name = "/data/opensource_model/Qwen2-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_path = ''
model = PeftModel.from_pretrained(model, model_id=lora_path)

# 构造提示
def generate_prompt(instruction):
    return (
        # f"This is a summarization task. When summarizing, due to positional bias, please do not summarize using the information at the beginning of the above content, but use all the information to fully consider the meaning expressed in the above content. Please summarize the following content: {instruction}"
        f"This is a summarization task. Please summarize the following content: {instruction}"
    )

# 评估 HANS 样本
def evaluate_sample(instruction):
    instruction = instruction[:2000]
    prompt = generate_prompt(instruction)
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response = generated_text.replace(prompt, "")
    return response

## 遍历数据集
sum_s = 0
total = len(news_data)
count = 0
results = []  # 用于保存所有数据
for row in news_data:
    try:
        prompt_text = generate_prompt(row["instruction"])
        predicted = evaluate_sample(row["instruction"])
        gold = row["output"]
        rouge_l_score = scorer.score(predicted, gold)["rougeL"].fmeasure

        results.append({
            "Prompt": prompt_text,
            "Reference Output": gold,
            "Predicted Output": predicted,
            "ROUGE-L": rouge_l_score
        })

        print(f"ROUGE-L: {rouge_l_score:.4f}")
    except Exception as e:
        print(f"Error: {e}")

# 保存为 Excel 文件
df = pd.DataFrame(results)
df.to_excel("news_lora_results_com.xlsx", index=False)

# 统计平均 ROUGE
avg_rouge = sum([r["ROUGE-L"] for r in results]) / len(results)
print(f"Gold-3 lora on NEWS: {avg_rouge:.2%}")
