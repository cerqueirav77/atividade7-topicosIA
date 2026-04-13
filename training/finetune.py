import os
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

MODEL_NAME = "NousResearch/Llama-2-7b-hf"
TRAIN_FILE = "data/train.jsonl"
TEST_FILE  = "data/test.jsonl"
OUTPUT_DIR = "./results"
ADAPTER_DIR = "./llama2-python-adapter"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                      
    bnb_4bit_quant_type="nf4",              
    bnb_4bit_compute_dtype=torch.float16,   
    bnb_4bit_use_double_quant=False,
)

print("Carregando modelo base quantizado...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

print("Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


lora_config = LoraConfig(
    r=64,                     
    lora_alpha=16,             
    lora_dropout=0.1,         
    bias="none",
    task_type="CAUSAL_LM",    
)


print("Carregando dataset...")
dataset = load_dataset(
    "json",
    data_files={"train": TRAIN_FILE, "test": TEST_FILE},
)

def format_prompt(example):
    """Formata cada par (prompt, response) no template de instrução."""
    return {
        "text": f"### Instrução:\n{example['prompt']}\n\n### Resposta:\n{example['response']}"
    }

dataset = dataset.map(format_prompt)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",     
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,             
    group_by_length=True,
    lr_scheduler_type="cosine",    
    report_to="none",
)

print("Configurando SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

print("Iniciando treinamento...")
trainer.train()

print(f"Salvando modelo adaptador em '{ADAPTER_DIR}'...")
trainer.model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print("Fine-tuning concluído com sucesso!")