%%capture
#!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
#!pip install -- no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 
dtype = None 
load_in_4bit = True

fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",   
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", 
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",      
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",         
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            
] 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",   
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

from datasets import load_dataset
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
trainer_stats = trainer.train()
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

FastLanguageModel.for_inference(model) 
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", 
        "1, 1, 2, 3, 5, 8", 
        "", 
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
FastLanguageModel.for_inference(model) 
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonnaci sequence.", 
        "1, 1, 2, 3, 5, 8",
        "", 
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

import json
from unsloth import FastLanguageModel
from transformers import TextStreamer
from datetime import datetime
import random
import requests 

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model", 
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
FastLanguageModel.for_inference(model) 

user_profile = {
    "user_id": "user123", 
    "performance": [],  
    "weak_topics": [],  
    "last_activity": str(datetime.now()),
}

def evaluate_answer(question, user_answer, correct_answer):
    if user_answer == correct_answer:
        user_profile["performance"].append({"question": question, "result": "correct", "time": str(datetime.now())})
        print("Correct!")
    else:
        user_profile["performance"].append({"question": question, "result": "incorrect", "time": str(datetime.now())})
        user_profile["weak_topics"].append(question["topic"])
        print("Incorrect. The correct answer is:", correct_answer)

def generate_personalized_question():
    if user_profile["weak_topics"]:
        topic = random.choice(user_profile["weak_topics"])
        print(f"Generating a question related to your weak topic: {topic}")
        alpaca_prompt = f"Generate a question on {topic}."
    else:
        alpaca_prompt = "Generate a general knowledge question."

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "",  
                "",  
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128)

def suggest_resources():
    if user_profile["weak_topics"]:
        for topic in set(user_profile["weak_topics"]):
            print(f"Suggested resources for {topic}:")
            try:
                response = requests.get(f"https://api.example.com/search?query={topic}")
                resources = response.json()
                for i, resource in enumerate(resources[:3]):  
                    print(f"{i + 1}. {resource['title']} - {resource['url']}")
            except Exception as e:
                print(f"Error fetching resources for {topic}: {e}")
    else:
        print("No weak topics identified yet. Complete some questions to get personalized suggestions.")

def assignment_copilot(assignment):
    print(f"Assisting with assignment: {assignment['title']}")
    print(f"Generating hints and resources to help you complete this assignment...")

    alpaca_prompt = f"Provide detailed steps to solve the following assignment: {assignment['description']}"
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "", 
                "", 
            )
        ],
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=256) 

    suggest_resources()

question = {
    "text": "What is a famous tall tower in Paris?",
    "correct_answer": "Eiffel Tower",
    "topic": "Geography"
}
user_answer = "Eiffel Tower" 
evaluate_answer(question, user_answer, question["correct_answer"])

generate_personalized_question()
suggest_resources()

assignment = {
    "title": "History of the Eiffel Tower",
    "description": "Write an essay about the history, construction, and significance of the Eiffel Tower."
}
assignment_copilot(assignment)

with open(f"{user_profile['user_id']}_profile.json", "w") as outfile:
    json.dump(user_profile, outfile)

if False:
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        "lora_model", 
        load_in_4bit = load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained("lora_model")
    
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")

# Save to 8bit Q8_0
if False: model.save_pretrained_gguf("model", tokenizer,)
# Remember to go to https://huggingface.co/settings/tokens for a token!
# And change hf to your username!
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# Save to 16bit GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# Save to q4_k_m GGUF
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# Save to multiple GGUF options - much faster if you want multiple!
if False:
    model.push_to_hub_gguf(
        "hf/model", # Change hf to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "", # Get a token at https://huggingface.co/settings/tokens
    )