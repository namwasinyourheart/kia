import streamlit as st
import logging
# Configure root logger
logging.basicConfig(level=logging.INFO)
from agents.agent_graph import build_agent_graph
from agents.websearch import WebSearchAgent
from agents.chat import ChatAgent
from datetime import datetime
import pytz
import time

# Function to generate assistant's response message with streaming effect
def generate_response_message(response):
    full_response = ""
    response_words = response.split()
    with st.chat_message("Kia", avatar="🤖"):
        message_placeholder = st.empty()
        for word in response_words:
            full_response += word + " "
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.05)
        message_placeholder.markdown(full_response)
    return full_response

# Function to generate initial message
def generate_initial_message():
    vietnam_tz = pytz.timezone("Asia/Ho_Chi_Minh")
    current_time = datetime.now(vietnam_tz).time()
    if 5 <= current_time.hour < 12:
        greeting = "Good morning"
    elif 12 <= current_time.hour < 18:
        greeting = "Good afternoon"
    elif 18 <= current_time.hour < 21:
        greeting = "Good evening"
    else:
        greeting = "Hello"
    initial_prompt = f"{greeting}! How can I assist you?"
    return initial_prompt

st.set_page_config(
    page_title="KIA — Your Know-It-All Companion",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)
st.markdown(
    "<h1 style='white-space:nowrap;'>🤖 KIA — Your Know-It-All Companion</h1>",
    unsafe_allow_html=True
)

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Settings")

    # st.sidebar.header("Agent Settings")

    # Sidebar toggles for search agents
    if 'enable_docsearch' not in st.session_state:
            st.session_state['enable_docsearch'] = True
    if 'enable_websearch' not in st.session_state:
        st.session_state['enable_websearch'] = True
    with st.expander("Agent Settings", expanded=True):
        
        st.checkbox("Enable Document Search", value=st.session_state['enable_docsearch'], key='enable_docsearch')
        st.checkbox("Enable Web Search", value=st.session_state['enable_websearch'], key='enable_websearch')
    
    # Chat Controls
    with st.expander("💬 Chat Controls", expanded=True):
        if st.button("Clear Conversation"):
            st.session_state['chat_history'] = []
            st.rerun()
        # st.selectbox("Message Display", ["Expanded", "Compact"], index=0)

    # Settings Section
    with st.expander("🛠️ Display Settings", expanded=True):
        DEV_MODE = st.checkbox("Enable Dev Mode", value=False)
        STREAMING_ENABLED = st.checkbox("Enable response streaming", value=True)
        # # Theme selector
        # if 'theme' not in st.session_state:
        #     st.session_state.theme = "Light"
        # theme = st.selectbox("Theme", ["Light", "Dark"], index=["Light", "Dark"].index(st.session_state.theme))
        # if theme != st.session_state.theme:
        #     st.session_state.theme = theme
        #     # Apply theme
        #     if theme == "Dark":
        #         st.markdown("""
        #             <style>
        #                 .stApp {
        #                     background-color: #111;
        #                     color: #fff;
        #                 }
        #                 .stMarkdown, .stSelectbox, .stSlider {
        #                     color: #fff;
        #                 }
        #                 .stButton>button {
        #                     background-color: #333;
        #                     color: #fff;
        #                 }
        #                 .stExpander {
        #                     border-color: #333;
        #                 }
        #             </style>
        #         """, unsafe_allow_html=True)
    
    # # Agent Settings
    # with st.expander("🤖 Agent Settings", expanded=True):
    #     st.checkbox("Enable Web Search", value=True)
    #     st.checkbox("Enable Document Search", value=True)
    #     st.slider("Response Timeout (sec)", 10, 60, 30)
    
    # Information
    with st.expander("ℹ️ About", expanded=True):
        st.markdown("""### Kia Chatbot
        An intelligent assistant powered by LangGraph.
        
        **Quick Tips:**
        - Use clear, specific questions
        - Enable Dev Mode for detailed responses
        - Clear conversation for fresh context
        """)


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []


if 'waiting_for_response' not in st.session_state:
    st.session_state['waiting_for_response'] = False
if 'should_stream' not in st.session_state:
    st.session_state['should_stream'] = False
if 'initial_greeting_added' not in st.session_state:
    # Add initial greeting as persistent assistant message
    st.session_state['chat_history'].append({
        "user": "",
        "bot": generate_initial_message(),
        "agent": "chat",
        "rephrased_query": None,
        "trace": None,
        "websearch_results": None
    })
    st.session_state['initial_greeting_added'] = True
    st.session_state['should_stream'] = True

# Function that will store the user input in session state
def set_user_input():
    if st.session_state.user_message and st.session_state.user_message.strip():
        # Store the input in session state and mark that we're waiting for response
        st.session_state['waiting_for_response'] = True
        # Add user message to history immediately
        st.session_state['chat_history'].append({
            "user": st.session_state.user_message,
            "bot": None,  # Will be filled later
            "agent": None,
            "rephrased_query": None,
            "trace": None  # Ensure trace field exists but is None initially
        })
        # Force a rerun to show the user message immediately


# Display chat history in modern chat format



# Display all completed message pairs first
# Determine last completed assistant response index
# completed_idx = [idx for idx, t in enumerate(st.session_state['chat_history']) if t.get('bot') is not None]
# last_completed_idx = completed_idx[-1] if completed_idx else None

for i, turn in enumerate(st.session_state['chat_history']):
    # User message (skip empty greetings)
    if turn.get('user'):
        with st.chat_message("user", avatar="👤"):
            st.markdown(turn['user'])
    # Determine if this is the waiting placeholder
    is_waiting = st.session_state['waiting_for_response'] and i == len(st.session_state['chat_history']) - 1
    # Skip rendering for waiting placeholder or if no bot response yet
    if turn.get('bot') is None or is_waiting:
        continue
    # Assistant response rendering (streaming optional)
    if i == len(st.session_state['chat_history']) - 1 and STREAMING_ENABLED and st.session_state['should_stream']:
        generate_response_message(turn['bot'])
        st.session_state['should_stream'] = False
    else:
        with st.chat_message("Kia", avatar="🤖"):
            st.markdown(turn['bot'])
    # DEV_MODE trace expander (all completed turns)
    if DEV_MODE and turn.get('trace'):
        trace = turn.get('trace', [])
        # Group execution steps by agent
        agent_groups = {}
        for idx, step in enumerate(trace):
            ag = step.get('agent', 'unknown')
            agent_groups.setdefault(ag, []).append((idx, step))
        # Define agent display order
        agent_order = [
            ('orchestrator', 'OrchestratorAgent'),
            ('documentsearch', 'DocumentSearchAgent'),
            ('websearch', 'WebSearchAgent'),
            ('chat', 'ChatAgent'),
        ]
        # Show each agent's expander, even if no steps
        for ag_key, display_name in agent_order:
            steps = agent_groups.get(ag_key, [])
            with st.expander(f"{display_name} Steps", expanded=False):
                if not steps:
                    st.write("No steps executed.")
                for idx, step in steps:
                    st.markdown(f"**Step {idx+1}:**")
                    if isinstance(step, dict):
                        for k, v in step.items():
                            if k == 'context':
                                st.markdown(f"**{k}:**")
                                if isinstance(v, list):
                                    table_data = [[i+1, getattr(doc, 'page_content', str(doc)).strip()] for i, doc in enumerate(v)]
                                else:
                                    paragraphs = str(v).split('\n')
                                    table_data = [[i+1, p.strip()] for i, p in enumerate(paragraphs) if p.strip()]
                                if table_data:
                                    import pandas as pd
                                    df = pd.DataFrame(table_data, columns=["#", "Content"])
                                    st.dataframe(df, use_container_width=True,
                                                 column_config={"#": st.column_config.NumberColumn("#", width=50),
                                                               "Content": st.column_config.TextColumn("Content")})
                            elif k == 'prompt' or k == 'response' or k == 'classification':
                                st.markdown(f"**{k}:**")
                                st.code(str(v))
                            else:
                                st.markdown(f"**{k}:**")
                                st.code(str(v))
                    else:
                        st.code(str(step))

# Process message if waiting for response (AFTER displaying messages)
if st.session_state['waiting_for_response']:
    user_input = st.session_state['chat_history'][-1]['user']
    agent_graph = build_agent_graph()
    chat_history = st.session_state['chat_history'][:-1]  # Exclude the waiting message
    state = {
        "user_input": user_input,
        "chat_history": chat_history,
        "enable_docsearch": st.session_state['enable_docsearch'],
        "enable_websearch": st.session_state['enable_websearch'],
    }

    # --- Invoke agent graph (may take time) with error handling ---
    try:
        result_state = agent_graph.invoke(state)
    except Exception as e:
        # Use orchestrator fallback
        from agents.orchestrator import OrchestratorAgent
        error_result = OrchestratorAgent().handle_system_error(e)
        result_state = {
            'response': error_result['response'],
            'agent': error_result['agent'],
            'trace': error_result['trace'],
            'websearch_results': error_result['websearch_results']
        }
    response = result_state.get('response', '')
    agent_name = result_state.get('agent', result_state.get('next_agent', 'chat'))
    rephrased_query = None
    
    # Try to extract rephrased_query if available (for websearch agent)
    if agent_name == 'websearch':
        history_str = ""
        for turn in chat_history:
            if turn.get('bot'):  # Make sure there's a bot response in history
                history_str += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        from agents.websearch import WebSearchAgent
        rephrase_prompt = (
            f"Given the following conversation history:\n"
            f"{history_str}"
            f"User: {user_input}\n"
            f"Rephrase the user's latest question so it is clear and complete for a web search. "
            f"Only output the rephrased question."
        )
        rephrased_query = WebSearchAgent().llm.generate(rephrase_prompt).strip()
    
    # Update the last message with the bot's response
    st.session_state['chat_history'][-1].update({
        "bot": response,
        "agent": agent_name,
        "rephrased_query": rephrased_query,
        "trace": result_state.get('trace', []),
        "websearch_results": result_state.get('websearch_results', []) if agent_name == 'websearch' else None
    })
    
    # Mark next run to stream this new response
    st.session_state['should_stream'] = True
    
    # Reset waiting state
    st.session_state['waiting_for_response'] = False
    # Force rerun to show the completed response
    st.rerun()

# Modern chat UI using st.chat_input - place at the bottom
user_input = st.chat_input("Enter your message and press Enter...", key="user_message", on_submit=set_user_input)