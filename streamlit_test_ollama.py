# code by romer adapted to llama

import streamlit as st

# LLM section
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

# Loading/Ingestion section
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import ( DocxReader, PDFReader, PyMuPDFReader, ImageReader, PptxReader , FlatReader, HTMLTagReader )  # readers baseado no tipo de arquivo

# Indexing and Embedding section
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
try:
  from llama_index import VectorStoreIndex
except ImportError:
  from llama_index.core import VectorStoreIndex

# Storing section
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import os

st.set_page_config(page_title="Chat with the DB Knowledge Base, powered by LlamaIndex", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
#cohere.api_key = st.secrets.cohere_key
st.title("Chat with the DB Knowledge Base, powered by LlamaIndex 💬🦙")
st.info("Check out the full tutorial to build this app in our [blog post](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/)", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Database and Provisioning Documentation!"}
    ]

@st.cache_resource(show_spinner=False)
# Reader with `SimpleDirectoryReader`
def reader_and_loader_docs(load_file):
    _files = './data'
    global documents
    global index
    global vector_store
    global storage_context
    if load_file.endswith == 'pdf':
        parser = PDFReader()        
        file_extractor = {".pdf": parser}
        print(file_extractor)
    elif load_file.endswith == 'docx':
        parser = DocxReader()        
        file_extractor = {".docx": parser}
        print(file_extractor)
    elif load_file.endswith == '.jpg' or '.jpeg' or '.png':
        parser = ImageReader()
        file_extractor = {
            ".jpg": parser,
            ".jpeg": parser,
            ".png": parser
            }  # Add other image formats as needed
        print(file_extractor)
    elif load_file.endswith == 'pptx':
        parser = PptxReader()
        file_extractor = {".pptx": parser}
        print(file_extractor)
    elif load_file.endswith == 'html':
        parser = HTMLTagReader()
        file_extractor = {".html": parser}
        print(file_extractor)
    elif load_file.endswith == 'txt' or 'log':
        parser = FlatReader()
        file_extractor = {
            ".txt": parser, 
            ".log" : parser}
        print(file_extractor)
    else:
        print("Não existe loader para arquivo do tipo " + str(_files.endswith))
        exit()

    documents = SimpleDirectoryReader( _files, file_extractor=file_extractor ).load_data()
    
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)

def load_data():
    with st.spinner(text="Loading and indexing the SAPC docs – hang tight! This should take 1-2 minutes."):
        database_indexed_embedded_files = './database'
        _files = './data'
        for filename in os.listdir(_files):
            load_file = _files + filename
            reader = reader_and_loader_docs(load_file)
        docs = reader.load_data()
        llm = Ollama(model="llama3")
        # index = VectorStoreIndex.from_documents(docs)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
        
        Settings.embed_model = embed_model
        Settings.llm = llm
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        Settings.temperature = 0.5
        
        # initialize client
        db = chromadb.PersistentClient(path=database_indexed_embedded_files)

        # get collection
        chroma_collection = db.get_or_create_collection("quickstart")

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context, similarity_top_k=4)
                
        #index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

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
