import json
import os
from tqdm import tqdm
from prompt import SYSTEM_PROMPT_CODE, SYSTEM_PROMPT_EVAL
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

    # Generation parameters
    max_tokens = 2048

    # Load data
    with open('rcg/data/generated_data.json', 'r') as f:
        data = json.load(f)

    try:
        with open("rcg/data/generated_responses.json", "r") as f:
            responses = json.load(f)
    except:
        responses = []

    length = 0
    while 'better' in responses[length]:
        length += 1

    for i, item in enumerate(tqdm(data)):
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT_CODE},
            {"role": "user", "content": item['prompt']},
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=chat,
            temperature=0.1,
            max_completion_tokens=max_tokens
        )

        code_response = response.choices[0].message.content
        
        eval_prompt = f"User Request: {item['prompt']}\nCode: {code_response}"
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT_EVAL},
            {"role": "user", "content": eval_prompt}
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=chat,
            temperature=0.7,
            max_completion_tokens=max_tokens
        )

        eval_response = response.choices[0].message.content

        responses[i]['better'] = eval_response
        # responses[i+length]['better'] = eval_response

        # Save responses to file
        with open("rcg/data/generated_responses.json", "w") as f:
            json.dump(responses, f, indent=4)
