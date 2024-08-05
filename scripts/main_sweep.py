# %%
import huggingface_hub
import os
from datasets import load_dataset
import ollama
import pandas as pd
# %%
huggingface_hub.login(os.environ["HF_API_TOKEN"])


def load_eval_dataset(dataset_name="squad_v2"):
    # Load the SQuAD v2 dataset by default
    file_path = (
        "Meta-Llama-3.1-8B-evals/Details_squad_2024-07-22T14-58-08.291117.parquet.gzip"
    )
    dataset = load_dataset(dataset_name, data_files=file_path, split="train")
    return dataset


def generate_llm(prompt: str, model: str = "llama3.1", temperature=0.01) -> str:
    
    return ollama.generate(
        model=model,
        prompt=prompt,
        options=ollama.Options(temperature=temperature, num_predict=32),
    )["response"]


dataset = load_eval_dataset(dataset_name="meta-llama/Meta-Llama-3.1-8B-evals")

# %%
dataset

# %%
generate_llm("What is the capital of Texas?")


# %%
"{hi}".format(hi="hello")

# %%
import pickle
from collections import defaultdict
import time
from datasets.utils import tqdm

############################################
# Attempt 3:
############################################
predictions = defaultdict(list)
references = defaultdict(list)


eval_prompt = """ 
You are a evaluator that needs to compare if response1 matches any of the other_responses to determine if they are the same or not.

response1: {llm_response}

other_responses: {input_correct_responses}

Answer just "yes" or "no" to the question: "Do the responses match?"
"""


def evaluate_llm(temp=0.01, total_evals=1, model="llama3.1"):
    evaluation_history = []
    pbar = tqdm(total=total_evals)

    for i, example in enumerate(dataset):
        if i == total_evals:
            break
        generation = generate_llm(
            example["input_question"], model=model, temperature=temp
        )
        try:
            generation = generation.split("\n\n")[0]
        except:
            pass
        # print('START GENERATION:\n',  generation)
        # print('END GENERATION\n')
        # print('Correct responses:', example["input_correct_responses"])

        correct = generate_llm(
            eval_prompt.format(
                llm_response=generation,
                input_correct_responses=example["input_correct_responses"],
            ),
            model="llama3.1",
        )
        evaluation_history.append(
            {
                "id": example["input_question_hash"],
                "generation": generation,
                "input_correct_responses": example["input_correct_responses"],
                "correct": correct.lower().strip("."),
                "temperature": temp,
            }
        )
        pbar.update(1)
    # serialize to disk
    if '/' in model:
        model = model.split('/')[-1]
    with open(f"evaluation_history_{model=}_{temp=}_{total_evals=}.pkl", "wb") as f:
        pickle.dump(evaluation_history, f)
    return evaluation_history


# %%

all_evaluations = {}

# mistral text is 7B params
models = ["qwen2:7b-text", "mapler/gpt2"]

for model in models:
    for temp in [0.001, 0.25, 0.5, 0.75, 1.0, 1.5]:
        print('Evaluation for:', model, temp)
        print('Pulling model')
        ollama.pull(model)
        print('Pulled model')
        evaluation_history = evaluate_llm(temp, total_evals=1000, model=model)
        all_evaluations[f"{temp}_{model}"] = evaluation_history
        print(pd.DataFrame(evaluation_history)['correct'].value_counts())

# %%

# %%
all_evaluations
breakpoint()
