from graph import build_chat_graph

def main():
    print("🧠 MemoryBot is ready. Type 'exit' to quit.\n")

    chat_graph = build_chat_graph()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        # Execute graph with user input
        result = chat_graph.invoke({"user_input": user_input})

        answer = result.get("answer", "No response generated.")
        print(f"Bot: {answer}\n")

if __name__ == "__main__":
    main()
