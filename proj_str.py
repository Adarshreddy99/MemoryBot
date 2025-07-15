import os

# Define the folder structure
folders = [
    "teacher_chatbot/backend",
    "teacher_chatbot/frontend",
    "teacher_chatbot/models",
    "teacher_chatbot/storage"
]

# Define files with optional initial content
files = {
    "teacher_chatbot/backend/main.py": "# Entry point to run LangGraph flow\n",
    "teacher_chatbot/backend/memory.py": "# Chat memory + custom user memory logic\n",
    "teacher_chatbot/backend/rag.py": "# Corrective RAG logic (local + web fallback)\n",
    "teacher_chatbot/backend/load_model.py": "# Load GGUF LLM and MiniLM embedding model\n",
    "teacher_chatbot/backend/utils.py": "# Prompt formatting and keyword helpers\n",

    "teacher_chatbot/frontend/app.py": "# Streamlit or Flask frontend\n",

    "teacher_chatbot/storage/user_memory.json": "{}\n",  # Start as empty dict

    "teacher_chatbot/.env": "QDRANT_API_KEY=\nQDRANT_URL=\nCOLLECTION_NAME=teacher_chunks\n\nUPLOAD_DIR=storage/uploads\nEMBEDDINGS_DIR=storage/embeddings\nLOCAL_EMBED_MODEL_PATH=models/all-MiniLM-L6-v2\nLOCAL_MODAL_PATH=models/gemma-2b.Q4_K_M.gguf\n",

    "teacher_chatbot/requirements.txt": "# Add required packages like langchain, streamlit, llama-cpp-python, etc.\n",
    "teacher_chatbot/README.md": "# Teacher Chatbot Project\n\nSimple RAG + memory bot with LangChain, local models, and Qdrant.\n"
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files with initial content
for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print("✅ Project structure created at: teacher_chatbot/")
