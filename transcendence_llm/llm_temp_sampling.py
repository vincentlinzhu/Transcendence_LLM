from typing import List, Optional

import dataclasses
import os
import time
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


def evaluate_predictions(client, eval_dataset, squad_metric, temperatures=[0, 0.5, 1.0]):
    results = {}
    
    for temp in temperatures:
        predictions = []
        references = []

        for i, example in enumerate(eval_dataset):
            prompt = example['input_final_prompts'][0] # Prompt in string form
            reference_answers = example['output_parsed_answer'] # List of reference answers
            answer = get_model_prediction(client, prompt, temperature=temp)
            processed_answer = answer.split("\n\n")[0].strip()

            predictions.append({"id": example['id'], "prediction_text": processed_answer.lower()})
            references.append({"id": example['id'], "answers": {"text": reference_answers}})

            # Log progress
            if i % 50 == 0:
                print(f"Processed {i} examples at temperature {temp}")

            # Sleep to avoid hitting rate limits
            time.sleep(0.5)  # Adjust based on API rate limits

        # Compute metrics
        metric_result = squad_metric.compute(predictions=predictions, references=references)
        results[temp] = metric_result

    return results


def run_llm(cfg: Config):
    if cfg.wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity="project-eval",
            name=cfg.wandb_run_name,
            config=dataclasses.asdict(cfg),
        )
    
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))
    clinet = load_client(cfg)
    dataset = load_eval_dataset(dataset_name="meta-llama/Meta-Llama-3.1-8B-evals")

    # Load SQuAD v2 evaluation metric
    squad_metric = load("squad_v2")

    # Evaluate the model with different temperatures
    # evaluation_results = evaluate_predictions(dataset, model, tokenizer, squad_metric, temperatures=cfg.temperatures)
    evaluation_results = evaluate_predictions(clinet, dataset, squad_metric, temperatures=cfg.temperatures)
    for temp, metrics in evaluation_results.items():
        print(f"Temperature {temp}: F1 = {metrics['f1']}, EM = {metrics['exact_match']}")
        
    if cfg.wandb:
        wandb.finish()