from typing import List, Optional

import dataclasses
import os
import time
from tqdm import tqdm
from argparse import Namespace
from dataclasses import asdict, dataclass, field, make_dataclass
from functools import wraps
from importlib import metadata as importlib_metadata
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from eztils import abspath, datestr, setup_path, wlog
from eztils.argparser import HfArgumentParser, update_dataclass_defaults
from eztils.run_parallel import calculate_split, prod
from rich import print

import wandb
import torch
from transcendence_llm import Config
from huggingface_hub import InferenceClient, login
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from evaluate import load
from collections import defaultdict
import pickle
import requests
import numpy as np
import tiktoken

os.environ["token_num"] = "0"
token_num = os.environ["token_num"]
os.environ["token"] = os.getenv(f"HUGGINGFACE_HUB_TOKEN_{token_num}")

load_dotenv()


def load_eval_dataset(dataset_name="squad_v2"):
    # Load the SQuAD v2 dataset by default
    file_path = "Meta-Llama-3.1-8B-evals/Details_squad_2024-07-22T14-58-08.291117.parquet.gzip"
    dataset = load_dataset(dataset_name, data_files=file_path, split="train")
    return dataset


def load_client(cfg: Config):
    # Initialize the client for the Meta LLaMA 8B model
    client = InferenceClient(cfg.model_name)
    return client


def get_model_prediction(client, prompt, temperature=1.0, max_tokens=3500):
    messages = [{"role": "user", "content": prompt}]
    # try:
    response = client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    # Extract the response content
    answer = response.choices[0].message.content
    # answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    answer = answer.split("\n\n")[0].strip()

    # except Exception as e:
    #     print(f"Error during inference: {e}")
    #     answer = ""
    return answer
    
    
def get_model_prediction_API(model_api, prompt, temperature=1.0):
    # API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
    token = os.environ["token"]
    API_URL = model_api
    headers = {"Authorization": f"Bearer {token}"}
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Tokenize the input prompt
    input_tokens = tokenizer.encode(prompt)
    input_token_count = len(input_tokens)
    
    payload = {
        "inputs": prompt, 
        "options": {
            "wait_for_model": True
        },
        "parameters": {
            "temperature": temperature,  # Set your desired temperature here
            "max_new_tokens": 100
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        prediction = response.json()[0]['generated_text'].split(prompt)[1].split("\n\n")[0].strip()
        return prediction
    except Exception as e:
        os.environ["token_num"] = str(int(os.environ["token_num"]) + 1)
        if os.environ["token_num"] == "4":
            os.environ["token_num"] = "0"
        token_num = os.environ["token_num"]
        os.environ["token"] = os.getenv(f"HUGGINGFACE_HUB_TOKEN_{token_num}")
        return get_model_prediction_API(model_api, prompt, temperature)


def get_valid_samples(dataset):
    tokenizer = tiktoken.get_encoding("gpt2")
    valid_indices = []
    for i, example in enumerate(dataset):
        prompt = example['input_final_prompts'][0]
        input_tokens = tokenizer.encode(prompt)
        if len(input_tokens) < 900:
            valid_indices.append(i)
    return valid_indices


def generate_bootstrap_sample(dataset, sample_size):
    tokenizer = tiktoken.get_encoding("gpt2")
    # Identify indices where the prompt is less than 900 tokens
    valid_indices = []
    for i, example in enumerate(dataset):
        prompt = example['input_final_prompts'][0]
        input_tokens = tokenizer.encode(prompt)
        if len(input_tokens) < 900:
            valid_indices.append(i)

    # Ensure there are enough valid samples to draw from
    if len(valid_indices) < sample_size:
        raise ValueError("Not enough valid samples with prompts less than 900 tokens.")

    # Randomly sample from the valid indices
    random_sample_indices = np.random.choice(valid_indices, size=sample_size, replace=True)
    # Select and return the bootstrap sample
    return dataset.select(random_sample_indices)


def combine_all_temperature(cfg, bootstrap_sample):
    results = {}
    result_dict_0 = pickle.load(
        open(f"cached_eval_res_{cfg.model_name}_bootstrap_{bootstrap_sample}_temp_0.001.pkl", "rb")
    )
    result_dict_1 = pickle.load(
        open(f"cached_eval_res_{cfg.model_name}_bootstrap_{bootstrap_sample}_temp_0.3.pkl", "rb")
    )
    result_dict_2 = pickle.load(
        open(f"cached_eval_res_{cfg.model_name}_bootstrap_{bootstrap_sample}_temp_0.75.pkl", "rb")
    )
    result_dict_3 = pickle.load(
        open(f"cached_eval_res_{cfg.model_name}_bootstrap_{bootstrap_sample}_temp_1.5.pkl", "rb")
    )
    
    results.update(result_dict_0)
    results.update(result_dict_1)
    results.update(result_dict_2)
    results.update(result_dict_3)
    return results


# def evaluate_predictions(cfg, client, eval_dataset, squad_metric, temperatures=[0, 0.5, 1.0]):
def evaluate_predictions(bootstrap_sample, cfg, model_api, eval_dataset, squad_metric, temperatures=[0, 0.5, 1.0]):
    og_eval_dataset = eval_dataset
    results = defaultdict(list)
    
    # for bootstrap_sample in range(10):
    # eval_dataset = og_eval_dataset.select(range(250))
    eval_dataset = generate_bootstrap_sample(og_eval_dataset, 250)
    eval_dataset = generate_bootstrap_sample(og_eval_dataset, 1)
    for temp in temperatures:
        predictions = []
        references = []

        for i, example in tqdm(enumerate(eval_dataset), total=len(eval_dataset), desc="Processing"):
            prompt = example['input_final_prompts'][0] # Prompt in string form
            reference_answers = example['output_parsed_answer'] # List of reference answers
            
            # answer = get_model_prediction(client, prompt, temperature=temp)
            answer = get_model_prediction_API(model_api, prompt, temperature=temp)
            
            processed_answer = answer.lower()

            predictions.append({"id": str(i), "prediction_text": processed_answer, 'no_answer_probability': 0.0})
            references.append({"id": str(i), "answers": {"answer_start": [0], "text": reference_answers}})
            
            if i % 50 == 0:
                time.sleep(30)

        # Compute metrics
        metric_result = squad_metric.compute(predictions=predictions, references=references)
        results[temp] = metric_result
    
        time.sleep(60)
    
        with open(f"cached_eval_res_{cfg.model_name}_bootstrap_{bootstrap_sample}_temp_{temp}.pkl", "wb") as fin:
            pickle.dump(results, fin)
        
    return results


def plot_f1_scores(evaluation_results, cfg: Config):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    
    # Process Results
    formatted_data = defaultdict(list)
    for temp, metrics_list in evaluation_results.items():
        for metric in metrics_list:
            formatted_data[temp].append(metric['f1'])

    # Create a DataFrame with some data
    df = pd.DataFrame(formatted_data)
    print(df)
    
    # Mock Data:
    # data = {
    #     0.001: [4.5, 3.8, 4.4, 5.2, 4.7],
    #     0.5: [3.3, 3.1, 2.2, 3.2, 2.9],
    #     1: [2.1, 1.3, 2.8, 1.5, 1.9]
    # }
    # df = pd.DataFrame(data, index=[1, 2, 3, 4, 5])
    
    # Calculate mean and standard deviation
    means = df.mean()
    stds = df.std()
    sem = stds / np.sqrt(df.shape[0])
    # confidence_interval = 1.96 * sem


    # Create the x-axis labels from the DataFrame columns
    x_labels = df.columns
    title  ='Mean F1-Scores by Temperature'
    
    # Plotting
    plt.figure(figsize=(10, 6))
    # plt.bar(means.index, means, yerr=confidence_interval, capsize=5, color='skyblue', alpha=0.7)
    # plt.fill_between(means.index, means - confidence_interval, means + confidence_interval, color='gray', alpha=0.2, label='95% Confidence Interval')

    # plt.errorbar(x_labels, means, yerr=stds, fmt='-o', capsize=5, capthick=2, label=f'Model: {cfg.model_name}')
    
    plt.plot(x_labels, means, label=f'Model: {cfg.model_name}')
    plt.fill_between(x_labels, (means - sem), (means + sem), alpha=0.2)
    
    plt.title(f"{title}")
    plt.xlabel('Temperature')
    plt.ylabel('F1-Score')
    plt.xticks(x_labels)  # Ensure all x labels are shown
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}.png")


def run_llm(cfg: Config):
    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity="project-eval",
            name=cfg.wandb_run_name,
            config=dataclasses.asdict(cfg),
        )
    
    login(token=os.environ["token"])
    # client = load_client(cfg)
    dataset = load_eval_dataset(dataset_name="meta-llama/Meta-Llama-3.1-8B-evals")

    # Load SQuAD v2 evaluation metric
    squad_metric = load("squad_v2")

    
    # Save Evaluation Results:
    # if os.path.exists(f"cached_eval_res_{cfg.model_name}_3.pkl"):
    #     evaluation_results = pickle.load(
    #         open(f"cached_eval_res_{cfg.model_name}_3.pkl", "rb")
    #     )
    # else:
    
    evaluation_results = {}
    for bootstrap_sample in range(6, 10):
        evaluation_results = evaluate_predictions(bootstrap_sample, cfg, cfg.model_api, dataset, squad_metric, temperatures=cfg.temperatures)
    
    # evaluation_results = {}
    # for bootstrap_sample in range(250):
    #     evaluation_results = evaluate_predictions(bootstrap_sample, cfg, cfg.model_api, dataset, squad_metric, temperatures=cfg.temperatures)
    
    # evaluation_results = evaluate_predictions(cfg, client, dataset, squad_metric, temperatures=cfg.temperatures)
    
        # with open(f"cached_eval_res_{cfg.model_name}_3.pkl", "wb") as fin:
        #     pickle.dump(evaluation_results, fin) 
    
    # bootstrap_results = {}
    # for i in range(10):
    #     key = f"evaluation_results_{i}"
    #     # bootstrap_results[key] = pickle.load(
    #     #     open(f"cached_eval_res_{cfg.model_name}_bootstrap_{i}.pkl", "rb")
    #     # )
    #     bootstrap_results[key] = combine_all_temperature(cfg, i)
        
    # evaluation_results = defaultdict(list)
    # for key, results in bootstrap_results.items():
    #     for temp, f1_score in results.items():
    #         if temp not in evaluation_results:
    #             evaluation_results[temp] = [f1_score]
    #         else:
    #             evaluation_results[temp].append(f1_score)

    # plot_f1_scores(evaluation_results, cfg)  
    
    if cfg.wandb:
        wandb.finish()
        
        
        
 # evaluation_results = {}
    # evaluation_results_0 = pickle.load(
    #     open(f"cached_eval_res_{cfg.model_name}_0.pkl", "rb")
    # )
    # evaluation_results_1_2 = pickle.load(
    #     open(f"cached_eval_res_{cfg.model_name}_1_2.pkl", "rb")
    # )
    # evaluation_results_3 = pickle.load(
    #     open(f"cached_eval_res_{cfg.model_name}_3.pkl", "rb")
    # )
    # evaluation_results_4_5_6 = pickle.load(
    #     open(f"cached_eval_res_{cfg.model_name}_4_5_6.pkl", "rb")
    # )
    
    
    # for temp, metrics in evaluation_results_0.items():
    #     evaluation_results[temp] = metrics
    # for temp, metrics in evaluation_results_1_2.items():
    #     evaluation_results[temp] += metrics
    # for temp, metrics in evaluation_results_3.items():
    #     evaluation_results[temp] += metrics
    # for temp, metrics in evaluation_results_4_5_6.items():
    #     evaluation_results[temp] += metrics