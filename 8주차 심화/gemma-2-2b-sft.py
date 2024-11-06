import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers
import json

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

wandb.init(project='Hanghae99')
wandb.run.name = 'gemma2b-health'

load_dotenv()

hf_token = os.environ.get('HFToken')
login(hf_token)


with open('./disease_corpus.json', 'r', encoding='utf-8') as file:
    items = json.load(file)
    db = Dataset.from_dict({
        "question": [item["question"] for item in items],
        "answer": [item["answer"] for item in items]
    })

lora_config = LoraConfig(
    r=8,
    lora_alpha = 32,
    lora_dropout = 0.1,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

training_arguments = TrainingArguments(
    output_dir="/home/ubuntu/src/model/gemma_health_sft",
    logging_steps=100,
    eval_steps=100,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=4,
    logging_strategy="steps",
    eval_strategy="steps",
    num_train_epochs=2,
    fp16=True,
)

def generate_prompt(data):
    prompt_list = []
    for i in range(len(data['question'])):
        prompt_list.append(f"""<bos><start_of_turn>user
{data['question'][i]}<end_of_turn>
<start_of_turn>model
{data['answer'][i]}<end_of_turn><eos>""")
    return prompt_list


response_template = "<start_of_turn>model"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

train_split = db.train_test_split(test_size=0.2)
train_dataset, eval_dataset = train_split['train'], train_split['test']

trainer = SFTTrainer(
    max_seq_length=256,
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=generate_prompt,
    data_collator=collator,
    peft_config=lora_config,
)

checkpoint = None
last_checkpoint = get_last_checkpoint(training_arguments.output_dir)
if training_arguments.resume_from_checkpoint is not None:
    checkpoint = training_arguments.resume_from_checkpoint
else:
    checkpoint = last_checkpoint

train_result = trainer.train(resume_from_checkpoint=checkpoint)

trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()