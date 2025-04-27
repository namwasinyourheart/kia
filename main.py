from agents.agent_graph import build_agent_graph, visualize_agent_graph

def main():
    agent_graph = build_agent_graph()
    # Visualize the agent graph every time it is built
    visualize_agent_graph(agent_graph, as_image=True, save_to_file=True)
    chat_history = []
    print("Multi-Agent Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        state = {"user_input": user_input, "chat_history": chat_history}
        result_state = agent_graph.invoke(state)
        response = result_state.get('response', '')
        # Determine which agent produced the response, robust fallback
        if 'agent' in result_state:
            agent_label = result_state['agent']
        elif 'next_agent' in result_state:
            agent_label = result_state['next_agent']
        elif chat_history:
            agent_label = chat_history[-1].get('agent', 'chat')
        else:
            agent_label = 'chat'
        print(f"[{agent_label.capitalize()}Agent] Bot: {response}")
        # Display extra info for search agents
        if agent_label == 'websearch':
            results = result_state.get('websearch_results', [])
            print("Web Search Results:")
            for r in results:
                print(r)
        elif agent_label == 'documentsearch':
            context = result_state.get('context', [])
            print("Document Search Context:")
            for doc in context:
                print(getattr(doc, 'page_content', str(doc))[:200] + '...')
        print()
        # Update chat_history for next turn
        if 'chat_history' in result_state:
            chat_history = result_state['chat_history']
        else:
            chat_history.append({
                'user': user_input,
                'bot': response,
                'agent': agent_label,
                'trace': result_state.get('trace', []),
                'websearch_results': result_state.get('websearch_results', [])
            })

if __name__ == "__main__":
    main()
