from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "qwen2.5-3b-instruct"
)


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True
)


print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)


print("Model loaded successfully!")


messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    },
    {
        "role": "user",
        "content": "Explain artificial intelligence in simple words."
    }
]


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


model_inputs = tokenizer(
    [text],
    return_tensors="pt"
).to(model.device)


generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200
)


generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids
    in zip(
        model_inputs.input_ids,
        generated_ids
    )
]


response = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]


print("\n==============================")
print("LLM RESPONSE")
print("==============================")
print(response)