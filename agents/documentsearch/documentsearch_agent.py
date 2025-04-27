from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from agents.base.agent_base import AgentBase
from langgraph.graph import StateGraph, END
from agents.llms import LLM

def get_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        # output_dimensionality=768
    )
    return embeddings

class DocumentSearchAgent(AgentBase):
    def __init__(self, urls=None):
        # Crawl & index website content
        if urls is None:
            urls = ["https://en.thanglong.edu.vn/"]
        loader = WebBaseLoader(urls)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        embeddings = get_embeddings()
        self.vectorstore = FAISS.from_documents(docs, embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.graph = self.build_graph()

    def build_graph(self):
        def rephrase(state):
            # (Optional) Could implement query rephrasing using LLM here
            state.setdefault('trace', []).append({'step': 'rephrase', 'query': state['user_input']})
            return state
        def retrieve(state):
            query = state['user_input']
            results = self.retriever.invoke(query)
            # print("len(results)", len(results))
            # Store raw results instead of joining them
            state['context'] = results
            state['raw'] = results
            state.setdefault('trace', []).append({'step': 'retrieve', 'context': results})
            return state
        def synthesize(state):
            context = state.get('context', [])
            # print("len(context)", len(context))
            user_input = state['user_input']
            llm = LLM()

            # Format context as markdown table, one row per document
            table_rows = []
            for i, doc in enumerate(context, 1):
                # Escape pipe characters with HTML entity and replace newlines with spaces
                content = doc.page_content.replace('|', '&#124;').replace('\n', ' ').strip()
                table_rows.append(f"| {i} | {content} |")

            context_table = ""
            if table_rows:
                context_table = "| # | Content |\n|---|---------|\n" + "\n".join(table_rows)

            prompt = (
                f"Based on the following excerpts from https://en.thanglong.edu.vn/, answer the user's question as accurately as possible.\n\n"
                f"Excerpts (in markdown table format):\n{context_table}\n\n"
                f"User: {user_input}\nBot:"
            )
            response = llm.generate(prompt)
            state['response'] = response
            state.setdefault('trace', []).append({'step': 'synthesize', 'prompt': prompt, 'response': response})
            return state
        graph = StateGraph(dict)
        graph.add_node('rephrase', rephrase)
        graph.add_node('retrieve', retrieve)
        graph.add_node('synthesize', synthesize)
        graph.add_edge('rephrase', 'retrieve')
        graph.add_edge('retrieve', 'synthesize')
        graph.add_edge('synthesize', END)
        graph.set_entry_point('rephrase')
        return graph.compile()

    def run(self, query: str, chat_history=None):
        state = {'user_input': query, 'chat_history': chat_history or [], 'trace': []}
        result = self.graph.invoke(state)
        return result['response'], result['context'], result['trace']
