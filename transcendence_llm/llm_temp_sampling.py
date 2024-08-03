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
    try:
        response = client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        # Extract the response content
        answer = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    except Exception as e:
        print(f"Error during inference: {e}")
        answer = ""
    return answer
    
def get_model_prediction_API(model_api, prompt, temperature=1.0):
    # API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
    API_URL = model_api
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_HUB_TOKEN')}"}
    payload = {
        "inputs": prompt, 
        "parameters": {
            "temperature": temperature  # Set your desired temperature here
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text'].split(prompt)[1].split("\n\n")[0].strip()


# def evaluate_predictions(client, eval_dataset, squad_metric, temperatures=[0, 0.5, 1.0]):
def evaluate_predictions(model_api, eval_dataset, squad_metric, temperatures=[0, 0.5, 1.0]):
    results = defaultdict(list)
    
    for seed in range(10):
        eval_dataset = eval_dataset.select(range(seed*25, seed*25+25))
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
                    
                # Sleep to avoid hitting rate limits
                time.sleep(0.5)  # Adjust based on API rate limits

            # Compute metrics
            metric_result = squad_metric.compute(predictions=predictions, references=references)
            results[temp].append(metric_result)
        
            time.sleep(60)
        
    
    return results


def plot_f1_scores(evaluation_results, cfg: Config):
    import matplotlib.pyplot as plt
    import pandas as pd
    
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

    # Create the x-axis labels from the DataFrame columns
    x_labels = df.columns
    title  ='Mean F1-Scores by Temperature'
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(x_labels, means, yerr=stds, fmt='-o', capsize=5, capthick=2, label=f'Model: {cfg.model_name}')
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
    
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    # client = load_client(cfg)
    dataset = load_eval_dataset(dataset_name="meta-llama/Meta-Llama-3.1-8B-evals")

    # Load SQuAD v2 evaluation metric
    squad_metric = load("squad_v2")

    # evaluation_results = evaluate_predictions(dataset, model, tokenizer, squad_metric, temperatures=cfg.temperatures)
    # evaluation_results = evaluate_predictions(client, dataset, squad_metric, temperatures=cfg.temperatures)
    # evaluation_results = evaluate_predictions(cfg.model_api, dataset, squad_metric, temperatures=cfg.temperatures)
    # for temp, metrics in evaluation_results.items():
    #     print(f"Temperature {temp}: F1 = {metrics['f1']}, EM = {metrics['exact']}")
    
    # Save Evaluation Results:
    if os.path.exists(f"cached_eval_res_{cfg.model_name}.pkl"):
        evaluation_results = pickle.load(
            open(f"cached_eval_res_{cfg.model_name}.pkl", "rb")
        )
    else:
        evaluation_results = {}
        evaluation_results = evaluate_predictions(cfg.model_api, dataset, squad_metric, temperatures=cfg.temperatures)
        with open(f"cached_eval_res_{cfg.model_name}.pkl", "wb") as fin:
            pickle.dump(evaluation_results, fin)
        
    # evaluation_results = {}
    plot_f1_scores(evaluation_results, cfg)    
        
    if cfg.wandb:
        wandb.finish()