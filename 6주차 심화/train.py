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
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers.trainer_utils import get_last_checkpoint

wandb.init(project='Hanghae99')
wandb.run.name = 'gpt2-sft'

load_dotenv()

hf_token = os.environ.get('HFToken')
login(hf_token)


with open('./corpus.json', 'r', encoding='utf-8') as file:
    list = json.load(file)
    db = Dataset.from_dict({
        "question": [item["question"] for item in list],
        "answer": [item["answer"] for item in list]
    })

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

training_arguments = TrainingArguments(
    output_dir="/tmp/gpt2_sft_result",
    save_total_limit=1,
    logging_steps=300,
    eval_steps=300,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_strategy="steps",
    eval_strategy="steps",
    overwrite_output_dir=True,
    num_train_epochs=1
)

def generate_prompt(data):
    prompt_list = []
    for i in range(len(data['question'])):
        text = f"### Question: {data['question'][i]}\n ### Answer: {data['answer'][i]}"
        prompt_list.append(text)
    return prompt_list

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

train_split = db.train_test_split(test_size=0.2)
train_dataset, eval_dataset = train_split['train'], train_split['test']

trainer = SFTTrainer(
    max_seq_length= 512,
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=generate_prompt,
    data_collator=collator,
)

checkpoint = None
last_checkpoint = get_last_checkpoint(training_arguments.output_dir)  # 만약 output_dir에 checkpoint가 남아있으면 이를 사용하고, 없으면 None이 return됩니다.
if training_arguments.resume_from_checkpoint is not None:  # output_dir이 아닌 다른 위치에서의 checkpoint를 resume_from_checkpoint로 지정할 수 있습니다.
    checkpoint = training_arguments.resume_from_checkpoint
else:  # 아니면 last_checkpoint로 checkpoint를 지정합니다.
    checkpoint = last_checkpoint

train_result = trainer.train(resume_from_checkpoint=checkpoint)

trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
