import json
import glob
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from trl import SFTTrainer

ds = {"prompt":[]}
for file in glob.glob("for_dataset/*.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        ds["prompt"].append(data["prompt"])

ds = Dataset.from_dict(ds)


# Split the dataset into a training set and a test set with a 70/30 ratio
ds = ds.train_test_split(test_size=0.15)

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0].replace(prompt, "")


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

from transformers import TrainingArguments

args = TrainingArguments(
  output_dir = "mistral_instruct_generation",
  #num_train_epochs=5,
  max_steps = 100, # comment out this line if you want to train in epochs
  per_device_train_batch_size = 4,
  warmup_steps = 0.03,
  logging_steps=10,
  save_strategy="epoch",
  #evaluation_strategy="epoch",
  evaluation_strategy="steps",
  eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
  learning_rate=2e-4,
  bf16=True,
  lr_scheduler_type='constant',
)
max_seq_length = 2048

trainer = SFTTrainer(
  model=model,
  peft_config=peft_config,
  max_seq_length=max_seq_length,
  tokenizer=tokenizer,
  packing=True,
  args=args,
  train_dataset=ds["train"],
  eval_dataset=ds["test"],
  dataset_text_field="prompt"
)

trainer.train()
trainer.save_model("mistral_function_calling_v0")