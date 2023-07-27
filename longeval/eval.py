import argparse
import os
import json
from tqdm import tqdm

import torch
import numpy as np

from fastchat.model import get_conversation_template
from utils import maybe_monkey_patch, get_output_dir, longeval_load_model, load_testcases, test_topics_one_sample, test_lines_one_sample 

import random
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from longchat.train.monkey_patch.llama_condense_monkey_patch import CondenseRotaryEmbedding

from evaluate_mmlu import evaluate

def longeval_test(model, tokenizer, output_dir, args):
    if args.task == "topics":
        for num_topics in [5, 10, 15, 20, 25]:
            print(f"************ Start testing {num_topics} topics per prompt ***********")
            avg_length = 0

            test_file = os.path.join(args.test_dir, f"topics/testcases/{num_topics}_topics.jsonl")
            output_file = os.path.join(output_dir, f"{num_topics}_response.txt")
            
            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                _, prompt_length, summary = test_topics_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)

            print(f"************ Finish testing {num_topics} topics per prompt with average prompt length {avg_length} ************")
            if args.eval_shortest_only:
                break
            
    elif args.task == "lines":
        for num_lines in [200, 300, 400, 500, 600, 680]:
            print(f"************ Start testing {num_lines} lines per LRT prompt ************")
            test_file = os.path.join(args.test_dir, f"lines/testcases/{num_lines}_lines.jsonl")
            
            output_file = os.path.join(output_dir, f"{num_lines}_response.txt")
            num_correct = 0
            avg_length = 0

            test_cases = load_testcases(test_file)
            for idx, test_case in tqdm(enumerate(test_cases)):
                correct, prompt_length, summary = test_lines_one_sample(model=model, tokenizer=tokenizer, test_case=test_case, output_file=output_file, idx=idx, args=args)
                avg_length += prompt_length / len(test_cases)
                num_correct += correct
            accuracy = num_correct / len(test_cases)

            with open(output_file, "a+") as f:
                f.write(f"Accuracy: {accuracy}")

            print(f"************ Finish testing {num_lines} lines per prompt with average prompt length {avg_length}, accuracy: {accuracy} ************")
            if args.eval_shortest_only:
                break
    else:
        print(f"Unsupported task: {args.task}")

def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        random_ratios=None,
        selected_tokens=None,
        return_unt=False
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        probs = 0
        mi = 0
        # print("----------------")
        for random_ratio in random_ratios:
            for m in self.modules():
                if isinstance(m, (CondenseRotaryEmbedding)):
                    m.set_ratio(random_ratio, self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            if selected_tokens is not None:
                logits = torch.stack([logits[..., tokenizer(token).input_ids[-1]] for token in selected_tokens], -1)
            # print(logits.softmax(-1)[:, -1].flatten())
            probs = probs + logits.softmax(-1) / float(len(random_ratios))
            mi += (logits.softmax(-1) * logits.log_softmax(-1)).sum(-1) / float(len(random_ratios))
        logits = probs.log()
        ent = - (logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)
        mi += ent

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            if return_unt:
                return (loss,) + output if loss is not None else output, mi, ent
            return (loss,) + output if loss is not None else output

        if return_unt:
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ), mi, ent
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True, help="model path")
    parser.add_argument("--task", type=str, required=True, help="Which evaluation task to use. currently support [topics, lines]")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
    parser.add_argument("--max_gpu_memory", type=int, default=40, help="max per gpu memory in GiB. A100 is 40 or 80.")
    parser.add_argument("--longchat_flash_attn", action='store_true', help="Only apply to longchat models. Whether to enable flash attention to save memory, but slower.")
    parser.add_argument("--longchat_ratio", type=float, default=8, help="Only apply to longchat models. Use ratio=8 for 16K context length model. Only ratio=8 is supported now.")
    parser.add_argument("--interpolation_type", type=str, default=None)
    parser.add_argument("--eval_shortest_only", action='store_true', default=0, help="Only eval the shortest case for illustration purpose")
    parser.add_argument("--test_dir", type=str, default="evaluation", help="Directory of the testcases")
    parser.add_argument("--random_ratios", type=float, default=[3.5, 4, 4.5], nargs='+')
    args = parser.parse_args()

    maybe_monkey_patch(args)
    output_dir = get_output_dir(args)

    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = lambda *input, **kwargs: forward(*input, **kwargs, random_ratios=args.random_ratios)
    model, tokenizer = longeval_load_model(args)

    print(type(model))
    
    # longeval_test(model, tokenizer, output_dir, args)
    evaluate(tokenizer, model)
