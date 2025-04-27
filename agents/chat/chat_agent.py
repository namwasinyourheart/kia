from agents.llms import LLM
from agents.base.agent_base import AgentBase
from langgraph.graph import StateGraph, END
from agents.tools import get_today, add, subtract, multiply, divide, sqrt, factorial
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import json

class ChatAgent(AgentBase):
    def __init__(self):
        self.llm = LLM()
        # Create a tool-calling agent via create_tool_calling_agent
        # llm_with_tools = self.llm.chat_model
        # PromptTemplate for tool-calling agent (must have input_variables)
        prompt = PromptTemplate(
            input_variables=["input"],
            template=(
                "You are an assistant specialized in Thang Long University information. Use the provided tools when appropriate.\n"
                "\nQuestion: {input}\n{agent_scratchpad}"
            )
        )
        # Bind date and math tools
        tools = [get_today, add, subtract, multiply, divide, sqrt, factorial]
        llm_with_tools = self.llm.chat_model.bind_tools(tools)
        agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        # Graph-based pipeline fallback
        self.graph = self.build_graph()

    def build_graph(self):
        def check_date(state):
            query = state['user_input'].lower()
            # If user asks for today's date
            if 'hôm nay' in query or 'today' in query:
                # Call the underlying function directly to get today's date
                response = get_today.func()
                state['response'] = response
                state['handled'] = True
                state.setdefault('trace', []).append({'step': 'get_today', 'response': response, 'agent': 'chat'})
            return state

        def math_node(state):
            import re, json
            query = state['user_input'].lower()
            # Addition
            m = re.match(r"(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)", query)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                result = add.func(json.dumps({"a": a, "b": b}))
                state['response'], state['handled'] = result, True
                state.setdefault('trace', []).append({'step': 'add', 'input': {"a": a, "b": b}, 'response': result, 'agent': 'chat'})
                return state
            # Subtraction
            m = re.match(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", query)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                result = subtract.func(json.dumps({"a": a, "b": b}))
                state['response'], state['handled'] = result, True
                state.setdefault('trace', []).append({'step': 'subtract', 'input': {"a": a, "b": b}, 'response': result, 'agent': 'chat'})
                return state
            # Multiplication
            m = re.match(r"(\d+(?:\.\d+)?)\s*(?:\*|x|times)\s*(\d+(?:\.\d+)?)", query)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                result = multiply.func(json.dumps({"a": a, "b": b}))
                state['response'], state['handled'] = result, True
                state.setdefault('trace', []).append({'step': 'multiply', 'input': {"a": a, "b": b}, 'response': result, 'agent': 'chat'})
                return state
            # Division
            m = re.match(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", query)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                result = divide.func(json.dumps({"a": a, "b": b}))
                state['response'], state['handled'] = result, True
                state.setdefault('trace', []).append({'step': 'divide', 'input': {"a": a, "b": b}, 'response': result, 'agent': 'chat'})
                return state
            # Factorial
            m = re.match(r"(\d+)\s*(?:!|factorial)", query)
            if m:
                n = int(m.group(1))
                result = factorial.func(json.dumps({"n": n}))
                state['response'], state['handled'] = result, True
                state.setdefault('trace', []).append({'step': 'factorial', 'input': {"n": n}, 'response': result, 'agent': 'chat'})
                return state
            # Square root
            m = re.match(r"sqrt(?:are)? of (\d+(?:\.\d+)?)", query)
            if m:
                x = float(m.group(1))
                result = sqrt.func(json.dumps({"x": x}))
                state['response'], state['handled'] = result, True
                state.setdefault('trace', []).append({'step': 'sqrt', 'input': {"x": x}, 'response': result, 'agent': 'chat'})
                return state
            return state

        def format_history(state):
            chat_history = state.get('chat_history', [])
            history_str = ""
            for turn in chat_history:
                history_str += f"User: {turn['user']}\nBot: {turn['bot']}\n"
            state['history_str'] = history_str
            state.setdefault('trace', []).append({'step': 'format_history', 'history_str': history_str})
            return state

        def build_prompt(state):
            user_input = state['user_input']
            history_str = state.get('history_str', '')
            prompt = (
                f"You are a helpful assistant. Here is the conversation so far:\n"
                f"{history_str}"
                f"User: {user_input}\nBot:"
            )
            state['prompt'] = prompt
            state.setdefault('trace', []).append({'step': 'build_prompt', 'prompt': prompt})
            return state

        def llm_call(state):
            # Call LLM with tools and detect function calls
            messages = [HumanMessage(content=state['prompt'])]
            # res = self.llm.chat_model.invoke(messages)
            res = self.agent_executor.invoke({"input": state['prompt']})
            output = res.get("output")
            actions = res.get("actions", [])
            state['response'] = output
            state['handled'] = True
            state['actions'] = actions
            state.setdefault('trace', []).append({'step': 'llm_call', 'response': output, 'actions': actions, 'agent': 'chat'})
            return state


            # # If LLM returned a function call, execute the tool
            # if getattr(res, 'tool_calls', None):
            #     tc = res.tool_calls[0]
            #     print("tc", tc)
            #     tool_map = {t.name: t for t in [get_today, add, subtract, multiply, divide, sqrt, factorial]}
            #     tool = tool_map.get(tc.name)
            #     if tool:
            #         output = tool.func(json.dumps(tc.args))
            #         state['response'] = output
            #         state['handled'] = True
            #         state.setdefault('trace', []).append({'step': tc.name, 'input': tc.args, 'response': output, 'agent': 'chat'})
            #         return state
            # # Fallback: text response
            # state['response'] = res.content
            # state.setdefault('trace', []).append({'step': 'llm_call', 'response': res.content})
            # return state

        graph = StateGraph(dict)
        graph.add_node('check_date', check_date)
        graph.add_node('math', math_node)
        graph.add_node('format_history', format_history)
        graph.add_node('build_prompt', build_prompt)
        graph.add_node('llm_call', llm_call)
        # Route date queries directly to END
        graph.add_conditional_edges(
            'check_date',
            lambda state: state.get('handled', False),
            {True: END, False: 'math'}
        )
        # Route math queries directly to END
        graph.add_conditional_edges(
            'math',
            lambda state: state.get('handled', False),
            {True: END, False: 'format_history'}
        )
        graph.add_edge('format_history', 'build_prompt')
        graph.add_edge('build_prompt', 'llm_call')
        graph.add_edge('llm_call', END)
        graph.set_entry_point('check_date')
        return graph.compile()

    def run(self, user_input: str, chat_history=None):
        # Try the tool-enabled agent executor first
        try:
            # Use invoke to capture function calls and args
            result = self.agent_executor.invoke({"input": user_input})
            output = result.get("output")
            actions = result.get("actions", [])
            return output, actions
        except Exception:
            # Fallback to the original graph-based flow
            state = {'user_input': user_input, 'chat_history': chat_history or [], 'trace': []}
            result = self.graph.invoke(state)
            return result['response'], result['trace']
