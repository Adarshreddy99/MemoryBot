import os
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv

from langchain.memory import ConversationBufferWindowMemory
from langchain_core.memory import BaseMemory
from langchain_core.runnables import RunnableConfig

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

from typing import ClassVar

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

load_dotenv()

# ------------------------------
# Constants
# ------------------------------
MEMORY_WINDOW_SIZE = 32
USER_MEMORY_PATH = os.path.join("storage", "user_memory.json")
CHAT_MEMORY_KEY = "chat_history"
MEMORY_TRIGGER_KEYWORDS = {
    "remember", "save", "am", "need", "important", "keep", "memory", "change"
}

# ------------------------------
# 1. Conversation Memory (LangChain)
# ------------------------------
chat_memory = ConversationBufferWindowMemory(
    memory_key=CHAT_MEMORY_KEY,
    return_messages=True,
    k=MEMORY_WINDOW_SIZE,
)

# ------------------------------
# 2. User Memory (Custom Class)
# ------------------------------

class UserMemory(BaseMemory):
    memory_key: str = "user_memory"
    file_path: str = USER_MEMORY_PATH
    lemmatizer : ClassVar =  WordNetLemmatizer()

    def _load_user_memory_dict(self) -> Dict:
        if not os.path.exists(self.file_path):
            return {}
        with open(self.file_path, "r") as f:
            return json.load(f)

    def _save_user_memory_dict(self, data: Dict):
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_user_memory(self) -> List[str]:
        return list(self._load_user_memory_dict().values())

    def update_user_memory(self, key: str, value: str):
        mem = self._load_user_memory_dict()
        mem[key] = value
        self._save_user_memory_dict(mem)

    def delete_user_memory(self, key: str):
        mem = self._load_user_memory_dict()
        if key in mem:
            del mem[key]
            self._save_user_memory_dict(mem)

    def auto_update_user_memory(self, message: str):
        lowered = message.lower()
        tokens = word_tokenize(lowered)
        lemmas = {self.lemmatizer.lemmatize(token) for token in tokens}
        if lemmas & MEMORY_TRIGGER_KEYWORDS:
            key = f"fact_{len(self._load_user_memory_dict()) + 1}"
            self.update_user_memory(key, message.strip())

    def load_memory_variables(self, inputs: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        user_facts = self.get_user_memory()
        memory_str = "\n".join(f"- {fact}" for fact in user_facts)
        return {self.memory_key: memory_str}

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        user_input = inputs.get("input", "")
        self.auto_update_user_memory(user_input)

    def clear(self) -> None:
        self._save_user_memory_dict({})


# ------------------------------
# 3. Combined Utility
# ------------------------------

def load_memory_variables() -> Dict:
    """Manually combines both memory types for non-LangChain use."""
    chat_msgs = chat_memory.load_memory_variables({}).get(CHAT_MEMORY_KEY, [])
    user_mem = UserMemory().get_user_memory()
    user_mem_str = "\n".join(f"- {fact}" for fact in user_mem)
    return {
        "chat_history": chat_msgs,
        "user_memory": user_mem_str
    }


def save_chat(user_input: str, ai_response: str):
    """Updates both chat memory and user memory (if applicable)."""
    chat_memory.save_context({"input": user_input}, {"output": ai_response})
    UserMemory().auto_update_user_memory(user_input)
