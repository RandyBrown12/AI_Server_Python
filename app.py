from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os 

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

torch.cuda.empty_cache() #Clear Cache

if not os.path.exists(os.getenv('local_model_path')):
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(os.getenv('model_id'), torch_dtype=torch.float16)  # Use float16
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('model_id'))

    # Save the model and tokenizer locally
    print("Saving model locally...")
    model.save_pretrained(os.getenv('local_model_path'))
    tokenizer.save_pretrained(os.getenv('local_model_path'))
else:
    # Load the model and tokenizer from local path
    print("Loading model from local path...")
    model = AutoModelForCausalLM.from_pretrained(os.getenv('local_model_path'), torch_dtype=torch.float16)  # Use float16
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('local_model_path'))

device = 0 if torch.cuda.is_available() else -1

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device,  # Pass device to use GPU if available
)

print("Server is ready to accept requests!")

@app.post("/", response_class=PlainTextResponse)
async def generate(request: Request):
    try:
        prompt = await request.body()
        prompt = prompt.decode()
        
        messages = [
            {"role": "system", "content": os.getenv("system_content")},
            {"role": "user", "content": prompt},
        ]

        outputs = model_pipeline(
            messages,
            max_new_tokens=256,
        )

        return outputs[0]["generated_text"][-1]['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
