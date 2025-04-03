# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ***

# CHALLENGE 1: CREATE A DATA SCIENCE EXPERT USING THE INTRODUCTION TO STATISTICAL LEARNING WITH PYTHON PDF

# DIFFICULTY: BEGINNER

# SPECIFIC ACTIONS:
#  1. USE PDF LOADER TO LOAD THE PDF AND PROCESS THE TEXT
#  2. CREATE A VECTOR DATABASE TO STORE KNOWLEDGE FROM THE BOOK'S PDF
#  3. CREATE A WEB APP THAT INCORPORATES Q&A AND CHAT MEMORY

from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

# from langchain_openai import OpenAIEmbeddings

import streamlit as st
import os
from tempfile import NamedTemporaryFile
import yaml


def load_and_summarize(file):
    
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        file_path = tmp.name
        
    try:
        loader = PyPDFLoader(file_path)
        document = loader.load()
                
        model = Ollama(
            model="llama3.1",
            temperature = 0
        )
        
        docs = text_splitter.split_documents(documents)
        
        CHUNK_ZIE = 1000
        # Recursive Character Splitter: Uses "smart" splitting, and recursively tries to split until text is small enough
        text_splitter_recursive = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap=100,
        )
        
        docs_recursive = text_splitter_recursive.split_documents(documents)
        embedding_function = OllamaEmbeddings(
            model="llama3"
        )
        vectorstore = Chroma.from_documents(
            docs, 
            embedding=embedding_function, 
            persist_directory="data/chroma_2.db"
        )
    finally:
        os.remove(file_path)

    return response['output_text']

# 2.0 STREAMLIT INTERFACE
st.title("PDF Earning Call Summarizer")

st.subheader("Upload a PDF Document")
uploaded_file = st.file_uploader("Choose a file", type="pdf")

if uploaded_file is not None:
    
    if st.button('Summarize Document'):
        with st.spinner('Summarizing...'):
            
            summary = load_and_summarize(uploaded_file)
            
            st.subheader('Summarization Result:')
            st.markdown(summary)
else:
    st.write("No file uploaded. Please upload PDF file to proceed")
