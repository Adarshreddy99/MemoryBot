# 🧠 MemoryBot (LangGraph + Gemma + Custom RAG)

A memory-first chatbot designed to **understand**, **remember**, and **adapt** to user instructions — powered by LangChain, LangGraph, and local language models. Unlike traditional chatbots, this assistant prioritizes long-term personalization through a robust memory system, and only uses RAG (Retrieval-Augmented Generation) **when absolutely necessary**.

---

## 🧠 Core Focus: Memory

### 🔁 1. Conversation Memory (LangChain-native)
- Maintains chat history using `ConversationBufferWindowMemory`.
- Ensures coherent multi-turn conversations.
- Automatically resets and trims context beyond the window size.

---

### 🧍 2. Instructional User Memory (Custom & Editable)

A specialized user memory module that:

- 📝 **Stores custom user instructions**, facts, and preferences (e.g., "I am vegetarian", "Always use Celsius").
- 🧠 **Auto-fills based on intent detection** from keywords like:
  - `remember`, `save`, `I am`, `store`, `important`, `note that`, `don't forget`, etc.
- ✏️ **Manually editable** via UI or commands (add, delete, overwrite).
- 🔁 **Persistent and always injected** into the prompt during answer generation.

> Example:
> - User: "Remember that I work in finance."
> - Bot auto-saves: `"User works in finance"` into user memory.
> - All future answers reflect this context.

---

## 🧠 Memory-First, RAG-Second Strategy

This bot only uses RAG if:

- The query cannot be answered from conversation memory or user memory.
- The input is an information-seeking prompt (e.g., "what is", "explain", "who was", etc.)

---

## 🔄 Conditional RAG Flow (Corrective RAG)

When RAG is triggered:

- ✅ **Query Decomposition** is performed using local Gemma model.
- ✅ Sub-questions are generated and dense chunks are retrieved from **Qdrant Cloud** using **MiniLM-L6-v2** embeddings.
- ✅ If the chunks have **low similarity scores**, a fallback kicks in.

### 🌐 Web Search Integration (Internal Tool)
- If local content isn’t good enough, the bot performs a **web search using DuckDuckGo** internally (via LangChain tool).
- Relevant web snippets are integrated into the answer.
- This happens seamlessly — **no external APIs or browser tools** are required.

> 🔧 The entire process is automated and uses LangChain’s internal DuckDuckGo tool module.

---

## 🧭 LangGraph-Based Flow Control

The conversation logic is orchestrated using **LangGraph**, with:

- A shared state object (memory, user input, system output)
- **Routing based on input type**:
  - 💬 If it's conversational or memory-relevant → memory-only node
  - 🔍 If it’s information-seeking → RAG (CRAG) node
- Branching is based on **prompt keyword analysis + confidence score**.

---

## 📦 Tech Stack

- **LangChain** for tools, memory, and RAG abstraction
- **LangGraph** for building a dynamic flowchart-based conversation engine
- **Qdrant Cloud** for storing vectorized chunks
- **SentenceTransformers (MiniLM-L6-v2)** for local embeddings
- **Gemma 2B (GGUF)** via `llama.cpp` for all generation and reasoning
- **DuckDuckGo Tool** (LangChain) for internal fallback search

---

