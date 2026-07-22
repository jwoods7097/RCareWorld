import csv
import os
import json
from datetime import datetime
from pathlib import Path

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

FLEX = True
INPUT_PER_1M = 1.25
CACHED_INPUT_PER_1M = 0.125
OUTPUT_PER_1M = 10.00

CSV_PATH = Path(__file__).parent.parent / "log" / "llm_usage.csv"

# ============================================================================
# LLM Object
# ============================================================================
class OpenAILLM:
    """LLM abstraction and configuration"""

    # OpenAI API settings (using custom Qwen3 API endpoint)
    API_KEY = os.getenv("OPENAI_API_KEY", "")

    # NOTE: BASE_URL typically ends with /v1
    BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    # Use models supportting function calling
    MODEL = os.getenv("OPENAI_MODEL", "gpt-5.1")

    # OpenAI client instance (v1.0+ API)
    _client = None
  
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
        cls._client = None  # Reset client to use new key

    @classmethod
    def set_base_url(cls, base_url: str):
        """Set OpenAI base URL."""
        cls.BASE_URL = base_url
        cls._client = None  # Reset client to use new URL
    
    @classmethod
    def set_model(cls, model: str):
        """Set model."""
        cls.MODEL = model

    @classmethod
    def init_pipeline(cls):
        """Get or create OpenAI client instance."""
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI package not available")

        if cls._client is None:
            cls._client = OpenAI(
                api_key=cls.API_KEY,
                base_url=cls.BASE_URL
            )

    def generate(self, prompt, memory=True):
        # Verify that model has been instantiated
        if self._client is None:
            raise RuntimeError("The model has not been initialized yet, run LLM.init_pipeline() first.")
        
        # Call model
        self.conversation_history.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.MODEL,
            messages=self.conversation_history,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning,
            service_tier="flex" if FLEX else "default"
        )

        usage = response.usage

        if usage is not None:
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0
            total_tokens = usage.total_tokens or 0

            cached_input_tokens = 0
            if usage.prompt_tokens_details:
                cached_input_tokens = usage.prompt_tokens_details.cached_tokens or 0

            reasoning_tokens = 0
            if usage.completion_tokens_details:
                reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0

            uncached_input_tokens = input_tokens - cached_input_tokens

            total_cost = (
                uncached_input_tokens / 1_000_000 * INPUT_PER_1M
                + cached_input_tokens / 1_000_000 * CACHED_INPUT_PER_1M
                + output_tokens / 1_000_000 * OUTPUT_PER_1M
            )
            if FLEX:
                total_cost *= 0.5  # Apply 50% discount for flex tier

            row = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "uncached_input_tokens": uncached_input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost,
            }

            file_exists = os.path.exists(CSV_PATH)

            with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())

                if not file_exists:
                    writer.writeheader()

                writer.writerow(row)

        assistant_message = response.choices[0].message.content

        if memory:
            # Add response to history
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
        else:
            # Remove user prompt from history
            self.conversation_history.pop()

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
