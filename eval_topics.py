import os
import json

import torch
import transformers
import numpy as np

#path = "longchat_7b_2048"
#path = "longchat_7b_4096"
#path = "longchat_7b_8192"
path = "llama-7B-hf"
output_dir = "evaluation/topics/predictions"

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_dir = os.path.join(output_dir, path)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

tokenizer = transformers.AutoTokenizer.from_pretrained(path, use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

model = transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cuda()

for num_topics in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    print(f"Start testing {num_topics} per prompt!")
    test_file = f"evaluation/topics/testcases/{num_topics}_topics.jsonl"

    output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
    
    with open(test_file, 'r') as json_file:
        json_list = list(json_file)

    for test_case in json_list:
        test_case = json.loads(test_case)
        prompt = test_case["prompt"]
        prompt_length = test_case["prompt_length"]
        topics = test_case["topics"]
        input = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(input.input_ids.cuda(), max_new_tokens=100, use_cache=True)[0]
        outputs = outputs[prompt_length:]
        summary = f"Label: {topics[0]}, Predict: {tokenizer.batch_decode([outputs], skip_special_tokens=True)}, --- INFO --- Topics: {topics}, Length: {prompt_length}"
        print(summary)
        with open(output_file, "a+") as f:
            f.write(summary)
            f.write("\n")