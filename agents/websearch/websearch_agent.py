from langchain_community.utilities import GoogleSerperAPIWrapper
from agents.llms import LLM
from agents.base.agent_base import AgentBase
from langgraph.graph import StateGraph, END

class WebSearchAgent(AgentBase):
    def __init__(self):
        self.search = GoogleSerperAPIWrapper()
        self.llm = LLM()
        self.graph = self.build_graph()

    def build_graph(self):
        def rephrase(state):
            chat_history = state.get('chat_history', [])
            history_str = ""
            for turn in chat_history:
                history_str += f"User: {turn['user']}\nBot: {turn['bot']}\n"
            state['history_str'] = history_str
            rephrase_prompt = (
                f"You are a helpful assistant specialized in Thang Long University information.\n"
                f"Given the following conversation history:\n"
                f"{history_str}"
                f"User: {state['user_input']}\n"
                f"Rephrase the user's latest question so it is clear and complete for a web search. "
                f"Only output the rephrased question."
            )
            rephrased_query = self.llm.generate(rephrase_prompt).strip()
            state['rephrase_prompt'] = rephrase_prompt
            state['rephrased_query'] = rephrased_query
            state.setdefault('trace', []).append({'step': 'rephrase', 'rephrase_prompt': rephrase_prompt, 'rephrased_query': rephrased_query})
            return state

        def search(state):
            rephrased_query = state['rephrased_query']
            # Perform search and extract organic results robustly
            raw = self.search.results(rephrased_query)
            results = []
            if isinstance(raw, dict):
                organic = raw.get('organic')
                if isinstance(organic, list):
                    results = organic
                elif isinstance(organic, dict):
                    # Some implementations nest items within 'results' or 'items'
                    possible = organic.get('results') or organic.get('items') or []
                    results = possible if isinstance(possible, list) else []
            elif isinstance(raw, list):
                results = raw
            # Save in state and trace
            state['results'] = results
            state.setdefault('trace', []).append({'step': 'search', 'raw_results': raw})
            return state

        def synthesize(state):
            history_str = state.get('history_str', '')
            user_input = state['user_input']
            rephrased_query = state['rephrased_query']
            # Use search results from state
            results_list = state.get('results', [])
            # Build markdown table rows
            table_rows = []
            for i, item in enumerate(results_list, 1):
                title = item.get('title', '').replace('|', '&#124;').replace('\n', ' ').strip()
                snippet = item.get('snippet', '').replace('|', '&#124;').replace('\n', ' ').strip()
                link = item.get('link', '').replace('|', '&#124;').strip()
                table_rows.append(f"| {i} | {title} | {snippet} | {link} |")
            results_table = ''
            if table_rows:
                results_table = (
                    "| # | Title | Snippet | Link |\n"
                    "|---|-------|---------|------|\n" + "\n".join(table_rows)
                )
            prompt = (
                f"You are a helpful assistant specialized in Thang Long University information. "
                f"Here is the conversation so far:\n{history_str}"
                f"User: {user_input}\n"
                f"Web search was performed with the question: \"{rephrased_query}\"\n\n"
                f"Web search results (in markdown table):\n{results_table}\n\n"
                f"Answer the user's question based on these results. Bot:"
            )
            summary = self.llm.generate(prompt)
            state['response'] = summary
            state.setdefault('trace', []).append({'step': 'synthesize', 'prompt': prompt, 'response': summary})
            return state

        graph = StateGraph(dict)
        graph.add_node('rephrase', rephrase)
        graph.add_node('search', search)
        graph.add_node('synthesize', synthesize)
        graph.add_edge('rephrase', 'search')
        graph.add_edge('search', 'synthesize')
        graph.add_edge('synthesize', END)
        graph.set_entry_point('rephrase')
        return graph.compile()

    def run(self, query: str, chat_history=None):
        state = {'user_input': query, 'chat_history': chat_history or [], 'trace': []}
        result = self.graph.invoke(state)
        return result['response'], result['results'], result['trace']
