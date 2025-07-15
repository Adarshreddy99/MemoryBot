import os
from dotenv import load_dotenv
from typing import Dict, Any
import re

from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable

from memory import load_memory_variables, save_chat
from rag import corrective_rag
from load_model import load_llm

load_dotenv()
llm = load_llm()

# ------------------------------
# Step 1: Define the shared state
# ------------------------------
GraphState = Dict[str, Any]

# ------------------------------
# Step 2: Node: Load memory
# ------------------------------
def load_memory_node(state: GraphState) -> GraphState:
    memory_vars = load_memory_variables()
    return {
        **state,
        **memory_vars
    }

# ------------------------------
# Step 3: Node: Should Use RAG
# ------------------------------
def should_use_rag_node(state: GraphState) -> GraphState:
    query = state["user_input"].lower()
    keywords = [
        r"\bwhat\b", r"\bwhat's\b", r"\bhow\b", r"\bwhen\b", r"\bwhere\b",
        r"\bdefine\b", r"\bexplain\b", r"\bexplaining\b", r"\bexamples?\b",
        r"\bgive me\b", r"\btell me\b", r"\bsummarize\b", r"\bsummarise\b",
        r"\bsearch\b", r"\blook up\b", r"\bfind out\b",
        r"\bdifference between\b", r"\bcompare\b",
        r"\binformation about\b", r"\bdetails about\b",
    ]
    if any(re.search(pattern, query) for pattern in keywords):
        return {**state, "route": "use_rag"}
    else:
        return {**state, "route": "use_memory"}

# ------------------------------
# Step 4: Conditional routing function
# ------------------------------
def route_question(state: GraphState) -> str:
    return state["route"]

# ------------------------------
# Step 5: Node: RAG Generation
# ------------------------------
def rag_node(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    answer = corrective_rag(user_input)
    return {
        **state,
        "answer": answer
    }

# ------------------------------
# Step 6: Node: Memory-Only Generation
# ------------------------------
def memory_only_node(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    chat_history = state.get("chat_history", [])
    user_memory = state.get("user_memory", "")

    # Format chat history cleanly - only show last few exchanges
    formatted_history = ""
    if chat_history:
        # Only show last 6 messages (3 exchanges) to avoid overwhelming the prompt
        recent_history = chat_history[-6:]
        for speaker, message in recent_history:
            formatted_history += f"{speaker}: {message}\n"

    prompt = f"""You are a helpful assistant. Respond naturally to the user's message.

{user_memory if user_memory.strip() else ""}

{formatted_history if formatted_history.strip() else ""}

User: {user_input}
Assistant:"""

    response = llm(prompt, max_tokens=300, temperature=0.7, stop=["User:", "Assistant:"])
    
    if isinstance(response, dict):
        answer = response.get("choices", [{}])[0].get("text", "").strip()
    else:
        answer = str(response).strip()

    # Clean up any unwanted repetition
    lines = answer.split('\n')
    clean_answer = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("User:") and not line.startswith("Assistant:"):
            clean_answer.append(line)
    
    final_answer = ' '.join(clean_answer) if clean_answer else "I'm here to help!"

    return {
        **state,
        "answer": final_answer
    }

# ------------------------------
# Step 7: Node: Save memory
# ------------------------------
def save_node(state: GraphState) -> GraphState:
    updated_history = save_chat(state["user_input"], state["answer"])
    return {
        **state,
        "chat_history": updated_history
    }


# ------------------------------
# Step 8: Build the Graph
# ------------------------------
def build_chat_graph() -> Runnable:
    workflow = StateGraph(GraphState)

    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("should_use_rag", should_use_rag_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("memory_only_node", memory_only_node)
    workflow.add_node("save_node", save_node)

    workflow.set_entry_point("load_memory")

    workflow.add_edge("load_memory", "should_use_rag")
    workflow.add_conditional_edges(
        "should_use_rag", route_question,
        {
            "use_rag": "rag_node",
            "use_memory": "memory_only_node",
        }
    )
    workflow.add_edge("rag_node", "save_node")
    workflow.add_edge("memory_only_node", "save_node")

    # ✅ Stop graph here
    workflow.set_finish_point("save_node")

    return workflow.compile()

# ------------------------------
# Step 9: Chat Interface (loop driven by user)
# ------------------------------
def run_chat_session():
    chat_graph = build_chat_graph()
    print("🧠 MemoryBot is ready. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye!")
            break
        if not user_input:
            continue

        input_state = {
            "user_input": user_input,
            "answer": "",
            "route": "",
        }

        result = chat_graph.invoke(input_state)
        print(f"\nBot: {result['answer']}")

def run_single_query(query: str) -> str:
    chat_graph = build_chat_graph()
    state = {
        "user_input": query,
        "answer": "",
        "chat_history": [],
        "user_memory": "",
        "route": "",
    }
    result = chat_graph.invoke(state)
    return result["answer"]

if __name__=="__main__":
    run_chat_session()