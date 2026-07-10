import csv
import os
import json
from datetime import datetime
from pathlib import Path
import requests

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    print("[Warning] OpenAI package not installed. Install with: pip install openai")   

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

FLEX = False
INPUT_PER_1M = 1.25
CACHED_INPUT_PER_1M = 0.125
OUTPUT_PER_1M = 10.00

CSV_PATH = Path(__file__).parent.parent / "log" / "llm_usage.csv"

# ============================================================================
# LLM Object
# ============================================================================
class OpenAILLM:
    """LLM abstraction and configuration"""

    # OpenAI API settings
    API_KEY = os.getenv("OPENAI_API_KEY", "")

    # NOTE: BASE_URL typically ends with /v1
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Use models supportting function calling
    MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")
    
    # OpenAI request headers
    headers = None
  
    def __init__(self, system_prompt="You are a helpful assistant.", temperature=0.7, max_tokens=2048, reasoning="none"):
        self.conversation_history = [{
            "role": "system",
            "content": system_prompt
        }]

        # Temperature and other generation params
        self.temperature = temperature
        self.max_tokens = max_tokens  # None = no limit
        self.reasoning = reasoning

    @classmethod
    def set_api_key(cls, api_key: str):
        """Set OpenAI API key."""
        cls.API_KEY = api_key
        cls.headers = None  # Reset client to use new key

    @classmethod
    def set_base_url(cls, base_url: str):
        """Set OpenAI base URL."""
        cls.BASE_URL = base_url
    
    @classmethod
    def set_model(cls, model: str):
        """Set model."""
        cls.MODEL = model

    @classmethod
    def init_pipeline(cls):
        """Get or create OpenAI client instance."""
        cls.headers = {
            "Authorization": f"Bearer {cls.API_KEY}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt, memory=True):
        # Verify that model has been instantiated
        if self.headers is None:
            raise RuntimeError("The model has not been initialized yet, run LLM.init_pipeline() first.")
        
        # Call model
        payload = {
            "action": "query",
            "request_source": "override_params",
            "model_provider": "openai",
            "model_name": self.MODEL,
            "query": prompt,
            "history": self.conversation_history,
            "model_params": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "thinking_level": self.reasoning.upper()
            },
            "response_format": {"type": "json"}
        }
        response = requests.post(self.BASE_URL, json=payload, headers=self.headers).json()

        # Compute usage statistics and log to CSV
        usage = response["metadata"].get("usage_metric", None)
        if usage is not None:
            # Get statistics
            row = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "input_tokens": usage["input_token_count"],
                "output_tokens": usage["output_token_count"],
                "total_tokens": usage["total_token_count"],
                "total_cost": usage["total_token_cost"],
            }

            # Write to CSV
            file_exists = os.path.exists(CSV_PATH)
            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)

        assistant_message = response["response"]

        # Add response to history
        if memory:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message
    
    def update_system_prompt(self, system_prompt):
        self.conversation_history[0] = {"role": "system", "content": system_prompt}
    
    def reset(self, system_prompt=None):
        if system_prompt is not None:
            self.conversation_history = [{
                "role": "system",
                "content": system_prompt
            }]
        else:
            self.conversation_history = [self.conversation_history[0]]

    def add_info(self, info):
        try:
            # Get index of last get_info call
            last_index = len(self.conversation_history) - 1 - self.conversation_history[::-1].index({"role": "user", "content": "Get the current scene information"})
            
            # Remove last get_info call and response from history
            self.conversation_history.pop(last_index)
            self.conversation_history.pop(last_index)
        except ValueError:
            pass
        
        # Add current get_info data to history
        self.conversation_history.append({"role": "user", "content": "Get the current scene information"})
        self.conversation_history.append({"role": "assistant", "content": json.dumps(info['data'], ensure_ascii=False)})
