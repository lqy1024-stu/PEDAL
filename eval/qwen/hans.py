import pandas as pd # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
import logging
from peft import PeftModel # type: ignore

# 加载 HANS 数据集
sampled_data = pd.read_csv("../dataset/syn/sampled_hans.txt", sep="\t")

# 加载 Llama 模型和分词器
model_name = "/data/opensource_model/Qwen2.5-14B"
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
    filename="log_prompt_en/qwen2.5/prom_lap_neg_hans.txt",  # 日志文件名
    level=logging.INFO,             # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
)
# 记录日志的函数
def log_message(message):
    logging.info(message)


# 构造提示
def generate_prompt(premise, conclusion):
    return (
        f"Given the premise '{premise}' and the hypothesis '{conclusion}', please classify the relationship between the premise and hypothesis, and answer only: neutral, entailment, or contradiction. Note that there are bias between the premise and hypothesis, such as overlap bias and negation bias. If there is an overlap bias, do not judge the relationship between premises and hypothesis based on the degree of word overlap between them."
        # f"Given the premise '{premise}' and the hypothesis '{conclusion}', please classify the relationship between the premise and hypothesis, and answer only: neutral, entailment, or contradiction. Note that there is a lexical overlap bias between the premise and hypothesis. Do not judge their relationship based on the degree of overlap of words in the premise and hypothesis, but rather judge their relationship based on semantics and logic."
        # f"Given the premise '{premise}' and the hypothesis '{conclusion}', please classify the relationship between the premise and hypothesis, and answer only: neutral, entailment, or contradiction. Note that there is a negative word bias between the premise and hypothesis, such as 'not', 'nobody', 'no', 'never', 'nothing', etc. Do not judge their relationship based on the presence of negative words in the premise or hypothesis, but rather judge their relationship based on semantics and logic."
    )

"""
# 构造提示
def generate_prompt(premise, conclusion):
    return (
        f"Given the premise '{premise}' and the hypothesis '{conclusion}', please classify the relationship between the premise and hypothesis, and answer only: neutral, entailment, or contradiction."
    )
"""

# 评估 HANS 样本
def evaluate_sample(premise, hypothesis, gold_label):
    prompt = generate_prompt(premise, hypothesis)
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
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
    log_message(f"generated_text: {generated_text}\n")
    # 提取生成的回答（去除提示部分）
    response = generated_text.replace(prompt, "")
    log_message(f"response: {response.lower()}\n")
    # 提取模型生成的答案
    if "entailment" == response.lower():
        predicted_label = "entailment"
    else:
        predicted_label = "non-entailment"
    return predicted_label == gold_label, predicted_label


subset_1 = sampled_data[sampled_data["heuristic"] == "lexical_overlap"]
entail_count = 0
non_entail_count = 0
entail_correct = 0
non_entail_correct = 0
print("lexical_overlap")
for _, row in subset_1.iterrows():
    if row["gold_label"] == "entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"entail_count: {entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        entail_count = entail_count + 1
        if is_correct:
            entail_correct += 1
    if row["gold_label"] == "non-entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"non_entail_count: {non_entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        non_entail_count = non_entail_count + 1
        if is_correct:
            non_entail_correct += 1
print(f"Accuracy for lexical_overlap entailment: {entail_correct / entail_count:.2%}")
log_message(f"Accuracy for lexical_overlap entailment: {entail_correct / entail_count:.2%}")
print(f"Accuracy for lexical_overlap non-entailment: {non_entail_correct / non_entail_count:.2%}")
log_message(f"Accuracy for lexical_overlap non-entailment: {non_entail_correct / non_entail_count:.2%}")


subset_2 = sampled_data[sampled_data["heuristic"] == "subsequence"]
entail_count = 0
non_entail_count = 0
entail_correct = 0
non_entail_correct = 0
print("subsequence")
for _, row in subset_2.iterrows():
    if row["gold_label"] == "entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"entail_count: {entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        entail_count = entail_count + 1
        if is_correct:
            entail_correct += 1
    if row["gold_label"] == "non-entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"non_entail_count: {non_entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        non_entail_count = non_entail_count + 1
        if is_correct:
            non_entail_correct += 1

print(f"Accuracy for subsequence entailment: {entail_correct / entail_count:.2%}")
log_message(f"Accuracy for subsequence entailment: {entail_correct / entail_count:.2%}")
print(f"Accuracy for subsequence non-entailment: {non_entail_correct / non_entail_count:.2%}")
log_message(f"Accuracy for subsequence non-entailment: {non_entail_correct / non_entail_count:.2%}")


subset_3 = sampled_data[sampled_data["heuristic"] == "constituent"]
entail_count = 0
non_entail_count = 0
entail_correct = 0
non_entail_correct = 0
print("constituent")
for _, row in subset_3.iterrows():
    if row["gold_label"] == "entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"entail_count: {entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        entail_count = entail_count + 1
        if is_correct:
            entail_correct += 1
    if row["gold_label"] == "non-entailment":
        is_correct, predicted = evaluate_sample(row["sentence1"], row["sentence2"], row["gold_label"])
        log_message(f"non_entail_count: {non_entail_count}, gold_label: {row['gold_label']}, predicted: {predicted}, is_correct: {is_correct}\n")
        non_entail_count = non_entail_count + 1
        if is_correct:
            non_entail_correct += 1
print(f"Accuracy for constituent entailment: {entail_correct / entail_count:.2%}")
log_message(f"Accuracy for constituent entailment: {entail_correct / entail_count:.2%}")
print(f"Accuracy for constituent non-entailment: {non_entail_correct / non_entail_count:.2%}")
log_message(f"Accuracy for constituent non-entailment: {non_entail_correct / non_entail_count:.2%}")
