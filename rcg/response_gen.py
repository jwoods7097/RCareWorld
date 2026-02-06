import torch
from transformers import pipeline
import json
import os
from tqdm import tqdm
from prompt import SYSTEM_PROMPT_EVAL
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    # OpenAI API settings (using custom Qwen3 API endpoint)
    API_KEY = os.getenv("OPENAI_API_KEY", "")

    # NOTE: BASE_URL typically ends with /v1
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # Use models supportting function calling
    MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

    # Set up OpenAI client
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    # Set up transformers pipeline
    pipe = pipeline(task="text-generation", model="Qwen/Qwen2.5-Coder-7B-Instruct", dtype=torch.bfloat16, device_map="auto")

    # Generation parameters
    temperature = 0.7
    max_tokens = 2048

    # Load data
    with open('rcg/data/generated_data.json', 'r') as f:
        data = json.load(f)

    try:
        with open("rcg/data/generated_responses.json", "r") as f:
            responses = json.load(f)
    except:
        responses = []

    length = len(responses)
    for item in tqdm(data[length:]):
        eval_prompt = f"User Request: {item['prompt']}\nCode: {item['incorrect']}"
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT_EVAL},
            {"role": "user", "content": eval_prompt}
        ]

        response = pipe(
            chat,
            temperature=temperature,
            max_new_tokens=max_tokens
        )
        torch.cuda.empty_cache()

        bad_response = response[0]["generated_text"][-1]["content"]

        response = client.chat.completions.create(
            model=MODEL,
            messages=chat,
            temperature=temperature,
            max_completion_tokens=max_tokens
        )

        good_response = response.choices[0].message.content

        responses.append({'bad': bad_response, 'good': good_response})

        # Save responses to file
        with open("rcg/data/generated_responses.json", "w") as f:
            json.dump(responses, f, indent=4)
