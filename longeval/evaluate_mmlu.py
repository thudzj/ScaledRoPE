import os
import torch
import numpy as np
import pandas as pd

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

choices = ["A", "B", "C", "D"]

def estimate_accs_of_bins_ranked_by_unc(is_correct, mis, num_bins=10):
    is_correct = torch.Tensor(is_correct)
    mis = torch.Tensor(mis)
    is_correct_ = is_correct[torch.argsort(mis)]
    is_correct_ = is_correct_[:is_correct_.shape[0] // num_bins * num_bins].view(num_bins, -1)
    return is_correct_.mean(-1) * 100.

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

@torch.no_grad()
def eval(ntrain, subject, model, tokenizer, dev_df, test_df, all_bins, all_bins2):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    mis = []
    ents = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]
        # print("label",label)

        output, mi, ent = model(
            input_ids=input_ids,
            selected_tokens=choices,
            return_unt=True
        )
        logits = output.logits[:,-1].flatten()
        mi = mi[:, -1].flatten()
        ent = ent[:, -1].flatten()

        mis.append(mi.item())
        ents.append(ent.item())
        # logits = torch.stack([logits[..., tokenizer(token).input_ids[-1]] for token in choices], -1)
        probs = logits.softmax(-1).cpu().to(torch.float32).numpy()
        # print("probs",probs)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    accs_of_bins_ranked_by_unc = estimate_accs_of_bins_ranked_by_unc(cors, mis)
    accs_of_bins_ranked_by_entrophy = estimate_accs_of_bins_ranked_by_unc(cors, ents)
    all_bins.append(accs_of_bins_ranked_by_unc)
    all_bins2.append(accs_of_bins_ranked_by_entrophy)
    print(accs_of_bins_ranked_by_unc, accs_of_bins_ranked_by_entrophy)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def evaluate(tokenizer, model, ntrain=5, data_dir="data", save_dir='mmlu_logs', save_name='none'):
    all_bins, all_bins2 = [], []
    model.eval()
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # if not os.path.exists(os.path.join(save_dir, "results_{}".format(save_name))):
    #     os.makedirs(os.path.join(save_dir, "results_{}".format(save_name)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in subjects:
        dev_df = pd.read_csv(
            os.path.join(data_dir, "dev", subject + "_dev.csv"), header=None
        )[: ntrain]
        test_df = pd.read_csv(
            os.path.join(data_dir, "test", subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(ntrain, subject, model, tokenizer, dev_df, test_df, all_bins, all_bins2)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # test_df["{}_correct".format(save_name)] = cors
        # for j in range(probs.shape[1]):
        #     choice = choices[j]
        #     test_df["{}_choice{}_probs".format(save_name, choice)] = probs[:, j]
        # test_df.to_csv(
        #     os.path.join(
        #         save_dir, "results_{}".format(save_name), "{}.csv".format(subject)
        #     ),
        #     index=None,
        # )

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # file = open('entrophy_res_lora.txt','a')
    # for i in range(len(all_bins)):
    #     s = str(all_bins[i]).replace('[','').replace(']','')
    #     s = s.replace("'",'').replace(',','') +'\n'  
    #     file.write(s)
    # file.close()
