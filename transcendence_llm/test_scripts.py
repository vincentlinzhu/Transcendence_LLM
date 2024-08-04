import pandas as pd
import tiktoken
from huggingface_hub import login
from datasets import load_dataset
from fuzzywuzzy import fuzz


def get_tokens_per_answer():
  # Load the dataset
  login("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")
  file_path = "Meta-Llama-3.1-8B-evals/Details_squad_2024-07-22T14-58-08.291117.parquet.gzip"
  dataset = load_dataset("meta-llama/Meta-Llama-3.1-8B-evals", data_files=file_path, split="train")

  # Initialize the tokenizer
  # Replace "gpt2" with your specific model's tokenizer name if needed
  tokenizer = tiktoken.get_encoding("gpt2")

  # Calculate the number of tokens in each answer
  total = 0
  for example in dataset:
    reference_answers = example['output_parsed_answer']
    total += len(tokenizer.encode(reference_answers[0]))

  avg_tokens = total/len(dataset)

  print(f"Average number of tokens in each answer: {avg_tokens}")


def test_fuzzywuzzy():
  # Define the strings
  string1 = "pope john paul ii"
  string2 = "john paul ii"

  # Calculate the partial ratio
  partial_ratio = fuzz.partial_ratio(string1, string2)
  print(f"The partial ratio between '{string1}' and '{string2}' is: {partial_ratio}")