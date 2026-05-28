import os
import json

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
            reasoning_effort=self.reasoning
        )

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
