import os
import subprocess
import sys

# Define the required packages
required_packages = [
    "pdfplumber"
]

# Check if packages are already installed
installed_packages = subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8")
packages_to_install = [package for package in required_packages if package not in installed_packages]

# Install required packages if they are not already installed
if packages_to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages_to_install)


import pathlib
import streamlit as st
from streamlit_chat import message
import openai
import PyPDF2
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from typing import List
from langchain.schema import Document
import pdfplumber

# Hide traceback
st.set_option('client.showErrorDetails', False)

# Setting page title and header
st.set_page_config(page_title="CODE CHAT", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center; color: red;'>PUBLIC POLICY CHATBOT</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Get to know the Public Policies at DxDy</h1>", unsafe_allow_html=True)

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

###############################################################################
# Get the OpenAI API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Set the OpenAI API key
openai.api_key = openai_api_key

################################################################### Using pdfplumber to read and extract pdf content
# Helper function to process PDFs using pdfplumber

def get_pdf_docs(folder_path):
    folder = pathlib.Path(folder_path)
    print("Iterating through PDF files")

    # Iterate over only .pdf files in the folder (including subdirectories)
    for pdf_file in folder.glob("**/*.pdf"):
        print(pdf_file)
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            yield Document(page_content=text, metadata={"source": str(pdf_file.relative_to(folder))})


# Use the Python code text splitter from Langchain to create chunks
def get_source_chunks(repo_path, pdf_folder_path): 
    source_chunks = []
    print("Creating source chunks")

    # Create a PythonCodeTextSplitter object for splitting the code
    splitter = PythonCodeTextSplitter(chunk_size=1024, chunk_overlap=30)

    # for source in get_repo_docs(repo_path):
    #     for chunk in splitter.split_text(source.page_content):
    #         source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

    for pdf in get_pdf_docs(pdf_folder_path):
        for chunk in splitter.split_text(pdf.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=pdf.metadata))
    return source_chunks

# Define function to generate response from user input
# This will also create the embeddings and store them in ChromaDB if it does not exist already
def generate_response(input_text):

    # Define the path of the repository and Chroma DB 
    # to get the absolute path of the current script file 
    REPO_PATH = os.path.abspath(os.path.dirname(__file__))
    CHROMA_DB_PATH = os.path.join(REPO_PATH, "chromaDB")

    vector_db = None

    # source_chunks = get_source_chunks(REPO_PATH, os.path.join(REPO_PATH, "sample_policies"))


    # Check if Chroma DB exists
    if not os.path.exists(CHROMA_DB_PATH):
        # Create a new Chroma DB
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        source_chunks = get_source_chunks(REPO_PATH, os.path.join(REPO_PATH, "sample_policies"))
        # Creating embeddings using the OpenAIEmbeddings, will incur costs
        vector_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        vector_db.persist()
    else:
        # Load an existing Chroma DB
        print(f'Loading Chroma DB from {CHROMA_DB_PATH}...')
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

    ############################################################################# Load a QA chain
    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vector_db.as_retriever())
    query_response = qa.run(input_text)

    # Example response object
    response = {
        "answer": query_response,  # Store the response text
        "metadata": {
            "source": "source_name"  # Add necessary metadata
        }
    }
    return response

# From here is the code for creating the chat bot using Streamlit and streamlit_chat
# container for chat history
response_container = st.container()

# container for text box
input_container = st.container()

with input_container:
    # Create a form for user input
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # If user submits input, generate response and store input and response in session state variables
        try:
            query_response = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(query_response)
        except Exception as e:
            st.error("An error occurred: {}".format(e))

###################################################### Updated accoding to generate_response function --> returning an object with meta
if st.session_state['generated']:
    # Display chat history in a container
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            response = st.session_state["generated"][i]
            st.code(response["answer"], language="python", line_numbers=False)
            st.text("Source: " + response["metadata"]["source"])
