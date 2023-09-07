import json
import torch
import time
import pandas as pd
from pathlib import Path
import os
import sys
from utils.dataset_utils import get_preprocessed_dataset
from configs.datasets import samsum_dataset, receipt_dataset

from ft_datasets.receipt_dataset import format


def evaluate(model, tokenizer, dataset_config):
    df_columns = ["file_name", "key", "gt", "prediction", "correct"]
    eval_df = pd.DataFrame(columns=df_columns)
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, 'val')
    model.eval()
    print("inside evaluate")
    print("dataset length: ", len(dataset))
    for i in tqdm(range(len(dataset))):
        sample_df = pd.DataFrame(columns=df_columns)
        ann = dataset.ann[i]
        correct = ann["output"]
        name = ann["fn"]

        sample_df["file_name"] = name
        eval_prompt = format.format_map(ann)
        
        # store all prediction time
        time_list = []
        with torch.no_grad():
            model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
            # measure how long prediction is taking
            start = time.time()
            out = tokenizer.decode(model.generate(**model_input, max_new_tokens=1000)[0], skip_special_tokens=True)
            end = time.time()
            time_list.append(end - start)

            # read output as json
            try:
                json_output = json.loads(out)
                json_correct = json.load(correct)
            except:
                print("failed to load json on sample: ", name)
                print("result; ", out)
                json_output = None
                json_correct = None

            if json_output is not None:
                for key in json_output:
                    sample_df["key"] = key
                    sample_df["gt"] = json_correct.get(key, None)
                    sample_df["prediction"] = json_output[key]
                    sample_df["correct"] = json_output[key] == json_correct.get(key, None)
                    eval_df = eval_df.append(sample_df, ignore_index=True)

    print("Average prediction time: ", sum(time_list)/len(time_list))
    print("Variance of prediction time: ", sum([(x - sum(time_list)/len(time_list))**2 for x in time_list])/len(time_list))

    return eval_df 