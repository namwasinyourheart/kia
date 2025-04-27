from agents.chat import ChatAgent
from agents.websearch import WebSearchAgent
from agents.llms import LLM

class OrchestratorAgent:
    def __init__(self):
        self.chat_agent = ChatAgent()
        self.websearch_agent = WebSearchAgent()
        self.chat_history = []
        # Use the same LLM as other agents for consistency
        self.llm = LLM()
        # System prompt for all agents
        self.system_prompt = (
            "You are an AI assistant specialized in Thang Long University information. "
            "Provide accurate, concise, and clear responses in a professional style. "
            "Respond in the user's language. "
            "Do not provide personal or sensitive data, refuse harmful or inappropriate requests, "
            "avoid off-topic responses (politely decline queries outside Thang Long University domain). "
        )

    def _decide_agent(self, user_input: str, chat_history=None) -> str:
        """
        Decide which agent should handle the user's latest message using a hybrid approach:
        1. Context-aware keyword matching
        2. LLM-based intent classification
        3. Conversation history analysis
        """
        # 1. Enhanced keyword matching with categories and weights
        keyword_groups = {
            # 'direct_reference': {
            #     'keywords': ['thanglong.edu.vn', 'trang web thang long', 'website trường'],
            #     'weight': 1.0
            # },
            'document_type': {
                'keywords': ['tài liệu', 'document', 'file', 'pdf', 'doc', 'văn bản'],
                'weight': 0.7
            },
            # 'information_intent': {
            #     'keywords': ['thông tin', 'tra cứu', 'tìm hiểu', 'xem', 'đọc'],
            #     'weight': 0.5
            # },
            # 'education_context': {
            #     'keywords': ['trường', 'đại học', 'khoa', 'ngành', 'môn học'],
            #     'weight': 0.3
            # }
        }

        # Calculate keyword match score
        score = 0.0
        lowered = user_input.lower()
        for group, data in keyword_groups.items():
            if any(kw in lowered for kw in data['keywords']):
                score += data['weight']
        
        # If strong keyword match, use document search
        if score >= 1.0:
            return 'documentsearch'

        # 2. LLM-based intent classification
        history_context = ""
        if chat_history:
            # Get last 3 turns for context
            recent_history = chat_history[-3:]
            for turn in recent_history:
                history_context += f"User: {turn['user']}\nBot: {turn['bot']}\n"

        # Format conversation history as markdown table
        table_rows = []
        if chat_history:
            for turn in recent_history:
                # Escape pipe characters by replacing with HTML entity
                user_msg = turn['user'].replace('|', '&#124;')
                bot_msg = turn['bot'].replace('|', '&#124;')
                table_rows.extend([
                    f"| User | {user_msg} |",
                    f"| Bot | {bot_msg} |"
                ])
        context_table = ""
        if table_rows:
            context_table = "| Role | Message |\n|------|---------|\n" + "\n".join(table_rows)

        prompt = (
            "As an AI assistant, analyze the user's query and conversation context to determine if it requires searching through Thang Long University's website documents.\n\n"
            "Context:\n"
            f"Recent conversation:\n{context_table}\n\n"
            f"Current query: {user_input}\n\n"
            "Consider these factors:\n"
            "1. Is the query specifically about Thang Long University?\n"
            "2. Does it require accessing structured document content?\n"
            "3. Is the information likely to be found in university documentation?\n"
            "4. Would document search be more reliable than web search for this query?\n\n"
            "Output ONLY 'documentsearch' if document search is most appropriate, or 'other' if not."
        )

        decision = self.llm.generate(prompt).strip().lower()
        
        # 3. Final decision with score influence
        if 'documentsearch' in decision or score >= 0.7:
            return 'documentsearch'
        # Otherwise, use LLM to decide between chat/websearch
        history_str = ""
        if chat_history:
            for turn in chat_history:
                history_str += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        # Format full conversation history as markdown table
        table_rows = []
        if chat_history:
            for turn in chat_history:
                # Escape pipe characters by replacing with HTML entity
                user_msg = turn['user'].replace('|', '&#124;')
                bot_msg = turn['bot'].replace('|', '&#124;')
                table_rows.extend([
                    f"| User | {user_msg} |",
                    f"| Bot | {bot_msg} |"
                ])
        history_table = ""
        if table_rows:
            history_table = "| Role | Message |\n|------|---------|\n" + "\n".join(table_rows)

        # Old routing prompt (commented out):
        # prompt = (
        #     "You are an orchestrator for a multi-agent assistant. "
        #     "Decide which agent should handle the user's latest message: "
        #     "output ONLY 'chat' (for general conversation, Q&A, reasoning, etc.) "
        #     "or 'websearch' (if the user is asking for real-time, factual, or web-based information.)\n\n"
        #     f"Conversation so far:\n{history_table}\n\n"
        #     f"Current query: {user_input}\n\n"
        #     "Which agent should handle this? (chat/websearch):"
        # )
        # decision = self.llm.generate(prompt).strip().lower()
        # return "websearch" if "websearch" in decision else "chat"
        # New custom prompt for Thang Long University:
        prompt = (
            "You are an orchestrator specialized in Thang Long University information. "
            "Route the user's query to the most suitable agent: "
            "'chat' for follow-up or general queries, "
            "'websearch' for real-time or external data, "
            "'documentsearch' for internal university documents. "
            f"Conversation history:\n{history_table}\n\n"
            f"User query: {user_input}\n\n"
            "Respond with ONLY the agent name (chat/websearch/documentsearch)."
        )
        decision = self.llm.generate(prompt).strip().lower()
        # Route based on LLM decision
        if "documentsearch" in decision:
            return "documentsearch"
        if "websearch" in decision:
            return "websearch"
        # Default to chat
        return "chat"

    def handle_system_error(self, error: Exception) -> dict:
        """
        Fallback when a system error occurs: logs the error and returns an 'oops' message.
        """
        import logging
        logging.error("System error in OrchestratorAgent: %s", error, exc_info=True)
        return {
            "response": "😞 Oops—something went wrong on our end. Please try again in a moment.",
            "agent": "chat",
            "trace": [],
            "websearch_results": []
        }

    def finalize_response(self, result_state: dict) -> dict:
        """
        Refine the routed agent's output using the system prompt.
        """
        raw = result_state.get('response', '')
        question = result_state.get('user_input', '')
        # Construct detailed refinement prompt
        prompt = (
            f"{self.system_prompt}\n\n"
            f"Original question: {question}\n"
            "Draft answer:\n"
            f"{raw}\n\n"
            "Please refine this answer: polite, professional, and in the user's language; focus solely on answering the question without extraneous remarks. "
            "Return only the final answer text."
        )
        refined = self.llm.generate(prompt).strip()
        result_state['response'] = refined
        # Label finalized output under orchestrator
        result_state['agent'] = 'orchestrator'
        result_state.setdefault('trace', []).append({
            'step': 'finalize',
            'prompt': prompt,
            'response': refined,
            'agent': 'orchestrator'
        })
        return result_state

    def validate_input(self, state: dict) -> dict:
        """
        Enforce guardrails before routing: handle empty or off-topic queries.
        """
        # initialize guard flag
        state['guard_failed'] = False
        user_input = state.get('user_input', '').strip()
        # Empty input
        if not user_input:
            state['guard_failed'] = True
            state['response'] = "😞 Sorry, I didn't catch that. Please ask a question about Thang Long University."
            state['agent'] = 'orchestrator'
            # Record validation failure
            state.setdefault('trace', []).append({'step': 'validate', 'reason': 'empty input', 'agent': 'orchestrator'})
            state['websearch_results'] = []
            return state
        # Off-topic detection via LLM with conversation context
        chat_history = state.get('chat_history', [])
        history_str = ""
        for turn in chat_history:
            history_str += f"User: {turn['user']}\nBot: {turn['bot']}\n"
        classification_prompt = (
            f"Given the following conversation history:\n{history_str}"
            "Classify the following query as ON_TOPIC or OFF_TOPIC. Domain: Thang Long University information.\n"
            f"Query: \"{user_input}\"\n"
            "General greetings are not considered off-topic."
            "Respond with exactly ON_TOPIC or OFF_TOPIC."
        )
        classification = self.llm.generate(classification_prompt).strip().upper()
        if classification != "ON_TOPIC":
            state['guard_failed'] = False
            state['response'] = "😞 Sorry, I can only answer questions about Thang Long University."
            state['agent'] = 'orchestrator'
            # Record off-topic validation
            state.setdefault('trace', []).append({'step': 'validate', 'reason': 'off topic_lm', 'agent': 'orchestrator'})
            state['websearch_results'] = []
            return state
        return state

    def evaluate_answer(self, state: dict) -> dict:
        """
        Evaluate if documentsearch response answers the user's question accurately.
        """
        response = state.get('response', '')
        question = state.get('user_input', '')
        context = state.get('context', [])
        prompt = (
            f"Given the question: \"{question}\" and the document search answer: \"{response}\", "
            "determine if the answer accurately addresses the question based on the provided context. "
            "If the provided text does not contain the information needed to answer the question, respond with NO. "
            "Respond with exactly YES or NO."
        )
        classification = self.llm.generate(prompt).strip().upper()
        answered = classification == "YES"
        # answered = False
        state['answered'] = answered
        # Record evaluation result
        state.setdefault('trace', []).append({
            'step': 'evaluate',
            'classification': classification,
            'agent': 'orchestrator'
        })
        return state

    # def route(self, user_input: str) -> str:
    #     agent_name = self._decide_agent(user_input, chat_history=self.chat_history)
    #     if agent_name == "websearch":
    #         result = self.websearch_agent.run(user_input, chat_history=self.chat_history)
    #     else:
    #         result = self.chat_agent.run(user_input, chat_history=self.chat_history)
    #     self.chat_history.append({"user": user_input, "bot": result})
    #     return result
