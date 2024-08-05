from typing import List, Optional
import dataclasses
import os
import time
from tqdm import tqdm
from argparse import Namespace
from dataclasses import asdict, dataclass, field, make_dataclass
from functools import wraps
from importlib import metadata as importlib_metadata
import pandas as pd
from dotenv import load_dotenv
from rich import print
import wandb
import torch
from transcendence_llm import Config
from huggingface_hub import InferenceClient, login
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from evaluate import load
from collections import defaultdict
import pickle
import requests
import numpy as np
import tiktoken
from fuzzywuzzy import fuzz

from transcendence_llm.utils import (
    load_eval_dataset,
    load_client,
    get_model_prediction,
    get_model_prediction_API,
    get_model_prediction_local,
    get_valid_samples,
    generate_bootstrap_sample,
    generate_sequential_sample,
    combine_all_temperature,
    evaluate_predictions,
    partial_f1_score,
    evaluate_model_output,
    visualize,
    plot_f1_scores
)

load_dotenv()

os.environ["token_num"] = "0"
token_num = os.environ["token_num"]
os.environ["token"] = os.getenv(f"HUGGINGFACE_HUB_TOKEN_{token_num}")



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
    
    # Load the pre-trained model and tokenizer from Hugging Face
    model_name = cfg.model_name  # Specify the model name
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(cfg.device)
    
    dataset = load_eval_dataset(dataset_name="meta-llama/Meta-Llama-3.1-8B-evals")
    squad_metric = load("squad_v2")
    evaluation_results = {}
    valid_indices = get_valid_samples(dataset)
    
    ############################################
    # Attempt 3:
    ############################################
    predictions = defaultdict(list)
    references = defaultdict(list)
    eval_dataset = generate_sequential_sample(dataset, 1000, valid_indices)
    for i, example in enumerate(tqdm(eval_dataset, desc="Processing examples")):
        for temp in cfg.temperatures:
            prompt = example['input_final_prompts'][0] # Prompt in string form
            reference_answers = example['output_parsed_answer'] # List of reference answers
            answer = get_model_prediction_local(model, tokenizer, cfg, prompt, temperature=temp)
            processed_answer = answer.lower()

            predictions[temp].append({"id": example['input_question_hash'], "prediction_text": processed_answer, 'no_answer_probability': 0.0})
            references[temp].append({"id": example['input_question_hash'], "answers": {"answer_start": [0], "text": reference_answers}})
        
        # Save the predictions anf references every 100 iterations
        if (i % 100 == 0 and i > 0) or i == 999:
            which_hundred = i // 100
            print(f"[bold red]Saving Iteration {i}")
            with open(f"sa_predictions_{cfg.model_name}_{which_hundred}.pkl", "wb") as fin:
                pickle.dump(predictions, fin)
            with open(f"sa_references_{cfg.model_name}_{which_hundred}.pkl", "wb") as fin:
                pickle.dump(references, fin)
            time.sleep(60)
            
    # f1_scores_by_temp = evaluate_model_output(cfg)
    # plot_f1_scores(f1_scores_by_temp, cfg)
    
    ############################################
    # Attempt 2:
    ############################################
    # for bootstrap_sample in range(1001):
    #     bootstrap_res = evaluate_predictions(valid_indices, bootstrap_sample, cfg, cfg.model_api, dataset, squad_metric, temperatures=cfg.temperatures)
    #     for temp, metrics in bootstrap_res.items():
    #         if temp not in evaluation_results:
    #             evaluation_results[temp] = [metrics]
    #         else:
    #             evaluation_results[temp].append(metrics)
                
    #     if bootstrap_sample % 100 == 0 and bootstrap_sample > 0:
    #         which_hundred = bootstrap_sample // 100
    #         # which_hundred += 8
    #         print(f"[bold red]Saving Iteration {bootstrap_sample}")
    #         with open(f"single_eval_res_{cfg.model_name}_{which_hundred}.pkl", "wb") as fin:
    #             pickle.dump(evaluation_results, fin)
    #         time.sleep(60)
    # visualize(cfg)
    
    ############################################
    # Attempt 1:
    ############################################
    # for bootstrap_sample in range(6, 10):
    #     evaluation_results = evaluate_predictions(bootstrap_sample, cfg, cfg.model_api, dataset, squad_metric, temperatures=cfg.temperatures)
    # bootstrap_results = {}
    # for i in range(7):
    #     key = f"evaluation_results_{i}"
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