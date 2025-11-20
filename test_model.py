
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse
import json
import math
from llama_attn_replace_sft import replace_llama_attn
import os

from icecream import ic as pprint
from tqdm import tqdm
from vllm import LLM, SamplingParams

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_and_merge_model(model_path, adapter_path, output_path="merged_model"):
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="right",
        use_fast=True,
    )
    
    replace_llama_attn(inference=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Handle special tokens
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # Load and merge LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    return output_path

def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the QLoRA adapter")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the formatted test set")
    parser.add_argument("--merged_path", type=str, default="merged_model", help="Path to save merged model")
    args = parser.parse_args()
    
    # Step 1: Merge LoRA adapter with base model
    print("Merging LoRA adapter with base model...")
    merged_model_path = load_and_merge_model(
        args.base_model, 
        args.adapter_path,
        args.merged_path
    )
    print(f"Merged model saved to: {merged_model_path}")
    
    # Step 2: Load merged model for testing
    print("Loading merged model for testing...")
    tokenizer = AutoTokenizer.from_pretrained(
        merged_model_path,
        padding_side="right",
        use_fast=True,
    )
    
    llm = LLM(
        model=merged_model_path,
        tensor_parallel_size=1,
    )

    # Load test data
    with open(args.data_path, "r") as f:
        data = json.load(f)
    prompts = [e["question"] for e in data]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=9216,
        stop_token_ids=[tokenizer.eos_token_id]
    )
    
    # Generate responses
    print("Generating responses...")
    response = llm.generate(prompts, sampling_params)
    all_generations = [output.outputs[0].text for output in response]
    
    # Save results
    with open("qwen_wo3.json", "w") as f:
        json.dump(all_generations, f, indent=4)
    print("Test results saved to test_results.json")
if __name__ == "__main__":
    test_model()
