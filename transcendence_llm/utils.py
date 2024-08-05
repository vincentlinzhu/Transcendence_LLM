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


def get_model_prediction_local(model, tokenizer, cfg, prompt, temperature=2.0, max_new_tokens=100):
    # Check if a GPU is available and move the model to the GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
   # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token if not set

    # Encode the prompt and create attention mask
    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded_input['input_ids'].to(cfg.device)
    attention_mask = encoded_input['attention_mask'].to(cfg.device)

    # Generate text with sampling enabled
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=100 + len(input_ids[0]),  # Add max_new_tokens to prompt length
            temperature=temperature,
            do_sample=True,  # Enable sampling to use temperature
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the generated text and strip the prompt
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    prediction = generated_text[len(prompt):].strip().split("\n\n")[0]
    
    return prediction


def get_valid_samples(dataset):
    tokenizer = tiktoken.get_encoding("gpt2")
    valid_indices = []
    for i, example in enumerate(dataset):
        prompt = example['input_final_prompts'][0]
        input_tokens = tokenizer.encode(prompt)
        if len(input_tokens) < 900:
            valid_indices.append(i)
    return valid_indices


def generate_bootstrap_sample(dataset, sample_size, valid_indices):
    # Ensure there are enough valid samples to draw from
    if len(valid_indices) < sample_size:
        raise ValueError("Not enough valid samples with prompts less than 900 tokens.")

    # Randomly sample from the valid indices
    random_sample_indices = np.random.choice(valid_indices, size=sample_size, replace=True)
    # Select and return the bootstrap sample
    return dataset.select(random_sample_indices)


def generate_sequential_sample(dataset, sample_size, valid_indices):
    # Ensure there are enough valid samples to draw from
    if len(valid_indices) < sample_size:
        raise ValueError("Not enough valid samples with prompts less than 900 tokens.")

    # Select the first `sample_size` indices from the valid indices
    sequential_sample_indices = valid_indices[:sample_size]
    # Select and return the sequential sample
    return dataset.select(sequential_sample_indices)


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


def evaluate_predictions(valid_indices, bootstrap_sample, cfg, model_api, eval_dataset, squad_metric, temperatures=[0, 0.5, 1.0]):
# def evaluate_predictions(cfg, client, eval_dataset, squad_metric, temperatures=[0, 0.5, 1.0]):
    og_eval_dataset = eval_dataset
    results = defaultdict(list)
    
    # for bootstrap_sample in range(10):
    # eval_dataset = og_eval_dataset.select(range(250))
    # eval_dataset = generate_bootstrap_sample(og_eval_dataset, 250)
    eval_dataset = generate_bootstrap_sample(og_eval_dataset, 1, valid_indices)
    for temp in temperatures:
        predictions = []
        references = []
        # eval_dataset should be size 1
        for i, example in tqdm(enumerate(eval_dataset), total=len(eval_dataset), desc="Processing"):
        
            prompt = example['input_final_prompts'][0] # Prompt in string form
            reference_answers = example['output_parsed_answer'] # List of reference answers
            
            # answer = get_model_prediction(client, prompt, temperature=temp)
            answer = get_model_prediction_API(model_api, prompt, temperature=temp)
            
            processed_answer = answer.lower()

            predictions.append({"id": example['input_question_hash'], "prediction_text": processed_answer, 'no_answer_probability': 0.0})
            references.append({"id": example['input_question_hash'], "answers": {"answer_start": [0], "text": reference_answers}})

        # Compute metrics
        metric_result = squad_metric.compute(predictions=predictions, references=references)
        results[temp] = metric_result
        
    return results


# Custom function to calculate partial match score using fuzzy matching
def partial_f1_score(prediction, reference): # prediction and reference are strings
    # Calculate token-based F1 score
    prediction_tokens = prediction.split()
    reference_tokens = reference.split()
    
    # Calculate overlap using fuzzy matching
    overlap = sum(fuzz.partial_ratio(pred_token, ref_token) > 80
                  for pred_token in prediction_tokens
                  for ref_token in reference_tokens)
    
    # Precision and Recall calculation
    precision = overlap / len(prediction_tokens) if prediction_tokens else 0
    recall = overlap / len(reference_tokens) if reference_tokens else 0
    
    if precision + recall == 0:
        return 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_model_output(cfg: Config):
    predictions = pickle.load(
        open(f"sa_predictions_{cfg.model_name}_10.pkl", "rb")
    )
    references = pickle.load(
        open(f"sa_references_{cfg.model_name}_10.pkl", "rb")
    )
    f1_scores_by_temp = defaultdict(list)
    for t in cfg.temperatures:
        for pred, ref in zip(predictions[t], references[t]):
            prediction_text = pred['prediction_text']
            reference_texts = ref['answers']['text']
            f1_scores = [partial_f1_score(prediction_text, ref_text) for ref_text in reference_texts]
            best_f1 = max(f1_scores)  # Choose the best F1 score among the reference answers
            f1_scores_by_temp[t].append(best_f1)
            
    return f1_scores_by_temp


def visualize(cfg: Config):
    bootstrap_results = {}
    bootstrap_results_10 = pickle.load(
        open(f"single_eval_res_{cfg.model_name}_8.pkl", "rb")
    )
    
    # bootstrap_results.update(bootstrap_results_200)
    bootstrap_results.update(bootstrap_results_10)

    plot_f1_scores(bootstrap_results, cfg)
    

def plot_f1_scores(evaluation_results, cfg: Config):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # df = pd.DataFrame(evaluation_results)
    # print(df)
    
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
    x_labels = df.columns.astype(str)
    title  ='LLM Partial Match: F1-Scores by Temperature'
    # title  ='F1-Scores by Temperature'
    
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