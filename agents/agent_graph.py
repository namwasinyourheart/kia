from langgraph.graph import StateGraph, END
from agents.chat import ChatAgent
from agents.websearch import WebSearchAgent
from agents.documentsearch import DocumentSearchAgent


def chat_node(state):
    print("[AgentGraph] Invoking ChatAgent")
    agent = ChatAgent()
    user_input = state['user_input']
    chat_history = state.get('chat_history', [])
    response, trace = agent.run(user_input, chat_history)
    state['response'] = response
    state['agent'] = 'chat'
    # Accumulate trace entries with agent label
    full_trace = state.setdefault('trace', [])
    for entry in trace:
        e = dict(entry)
        e['agent'] = 'chat'
        full_trace.append(e)
    state.setdefault('chat_history', []).append({"user": user_input, "bot": response, "agent": "chat", "trace": trace})
    return state

def documentsearch_node(state):
    print("[AgentGraph] Invoking DocumentSearchAgent")
    agent = DocumentSearchAgent()
    user_input = state['user_input']
    chat_history = state.get('chat_history', [])
    response, context, trace = agent.run(user_input, chat_history)
    state['response'] = response
    state['context'] = context
    state['agent'] = 'documentsearch'
    # Accumulate trace entries with agent label
    full_trace = state.setdefault('trace', [])
    for entry in trace:
        e = dict(entry)
        e['agent'] = 'documentsearch'
        full_trace.append(e)
    state.setdefault('chat_history', []).append({"user": user_input, "bot": response, "agent": "documentsearch", "trace": trace})
    return state

def websearch_node(state):
    print("[AgentGraph] Invoking WebSearchAgent")
    agent = WebSearchAgent()
    user_input = state['user_input']
    chat_history = state.get('chat_history', [])
    response, results_json, trace = agent.run(user_input, chat_history)
    # Compose rephrase prompt and rephrased query for trace
    history_str = ""
    for turn in chat_history:
        history_str += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    rephrase_prompt = (
        f"Given the following conversation history:\n"
        f"{history_str}"
        f"User: {user_input}\n"
        f"Rephrase the user's latest question so it is clear and complete for a web search. "
        f"Only output the rephrased question."
    )
    rephrased_query = agent.llm.generate(rephrase_prompt).strip()
    # Get actual web search results (now as JSON)
    response, results_json, trace = agent.run(user_input, chat_history)
    state['response'] = response
    state['agent'] = 'websearch'
    state['websearch_results'] = results_json
    # Accumulate trace entries with agent label
    full_trace = state.setdefault('trace', [])
    for entry in trace:
        e = dict(entry)
        e['agent'] = 'websearch'
        full_trace.append(e)
    state.setdefault('chat_history', []).append({"user": user_input, "bot": response, "agent": "websearch", "trace": trace, "websearch_results": results_json})
    return state

from agents.orchestrator import OrchestratorAgent
import logging

orchestrator = OrchestratorAgent()  # Singleton instance for routing

def finalize_node(state):
    """
    Finalize response using OrchestratorAgent.system_prompt.
    """
    return orchestrator.finalize_response(state)

def input_guard_node(state):
    """
    Enforce input guardrails before routing.
    """
    return orchestrator.validate_input(state)

def orchestrator_node(state):
    user_input = state['user_input']
    chat_history = state.get('chat_history', [])
    agent_name = orchestrator._decide_agent(user_input, chat_history=chat_history)
    # Respect sidebar toggles
    enable_doc = state.get('enable_docsearch', True)
    enable_web = state.get('enable_websearch', True)
    
    # If both search agents are disabled, default to chat
    if not enable_doc and not enable_web:
        agent_name = 'chat'
    if agent_name == 'documentsearch' and not enable_doc:
        # fallback to websearch if enabled, else chat
        agent_name = 'websearch' if enable_web else 'chat'
    elif agent_name == 'websearch' and not enable_web:
        # fallback to docsearch if enabled, else chat
        agent_name = 'documentsearch' if enable_doc else 'chat'
    state['next_agent'] = agent_name

    # Log routing decision
    logging.info(f"[AgentGraph] orchestrator routing: doc_enabled={enable_doc}, web_enabled={enable_web}, chosen={agent_name}")
    return state

def evaluate_node(state):
    """
    Evaluate documentsearch output against the user's question.
    """
    return orchestrator.evaluate_answer(state)

def build_agent_graph():
    """
    Build the multi-agent graph with orchestrator, chat, websearch, and documentsearch nodes.
    Orchestrator decides routing based on user input and chat history.
    """
    graph = StateGraph(dict)
    graph.add_node('input_guard', input_guard_node)
    # Conditional routing: if guard_failed, skip to finalize; else continue to orchestrator
    graph.add_conditional_edges(
        'input_guard',
        lambda state: state.get('guard_failed', False),
        {True: END, False: 'orchestrator'}
    )
    graph.add_node('orchestrator', orchestrator_node)
    graph.add_node('chat', chat_node)
    graph.add_node('websearch', websearch_node)
    graph.add_node('documentsearch', documentsearch_node)
    graph.add_node('finalize', finalize_node)
    graph.add_node('evaluate', evaluate_node)
    graph.add_conditional_edges(
        'orchestrator',
        lambda state: state['next_agent'],
        {
            'chat': 'chat',
            'websearch': 'websearch',
            'documentsearch': 'documentsearch',
        }
    )
    graph.add_edge('chat', 'finalize')
    graph.add_edge('websearch', 'finalize')
    graph.add_edge('documentsearch', 'evaluate')
    # Evaluate documentsearch and fallback based on search toggles
    graph.add_conditional_edges(
        'evaluate',
        lambda state: (
            'finalize' if state.get('answered', False)
            else 'websearch' if state.get('enable_websearch', True)
            # else 'documentsearch' if state.get('enable_docsearch', True)
            else 'chat'
        ),
        {
            'finalize': 'finalize',
            'websearch': 'websearch',
            'documentsearch': 'documentsearch',
            'chat': 'chat',
        }
    )
    graph.add_edge('finalize', END)
    graph.set_entry_point('input_guard')
    compiled_graph = graph.compile()
    return compiled_graph


def visualize_agent_graph(graph, as_image: bool = False, save_to_file: bool = False, file_path: str = "agent_graph"):
    """
    Visualize the compiled agent graph using Mermaid. Optionally render as PNG and/or save to file.
    """
    import os
    mermaid_code = graph.get_graph().draw_mermaid()
    if save_to_file and not as_image:
        with open(f"{file_path}.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_code)
        print(f"✅ Mermaid graph saved to: {file_path}.mmd")
    if as_image:
        try:
            png_data = graph.get_graph().draw_mermaid_png()
            if save_to_file:
                with open(f"{file_path}.png", "wb") as f:
                    f.write(png_data)
                print(f"✅ Mermaid graph image saved to: {file_path}.png")
            else:
                from IPython.display import Image, display
                display(Image(png_data))
        except Exception as e:
            print("❌ Failed to render PNG. Make sure `mermaid-render` is installed.")
            print(f"Error: {e}")
    else:
        print("\n--- Mermaid Graph ---\n")
        print(mermaid_code)
