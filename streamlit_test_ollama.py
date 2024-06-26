
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

from llama_index.readers.file import ( DocxReader, PDFReader, PyMuPDFReader, ImageReader, PptxReader , FlatReader, HTMLTagReader )

st.set_page_config(page_title="Chat with the DB Knowledge Base, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
#cohere.api_key = st.secrets.cohere_key
st.title("Chat with the DB Knowledge Base, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Database and Provisioning Documentation!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
        service_context = ServiceContext.from_defaults(llm=Ollama(model="lamma3", temperature=0.5, system_prompt="You are an expert on Core Network Telecomunnications and your job is to answer technical questions. Assume that all questions are related to Core Network equipments. Keep your answers technical and based on facts – do not hallucinate features."),
                                                    embed_model=embed_model)        
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="context")        

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
