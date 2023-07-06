import os
import subprocess
import sys
import pathlib
from flask import Flask, render_template, request
# from generate_chroma_db import generate_chroma_db

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


import pdfplumber
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Define the Flask app
app = Flask(__name__)

# # Get the OpenAI API key from environment variable
# openai_api_key = os.environ.get("OPENAI_API_KEY")

# # Set the OpenAI API key
# openai.api_key = openai_api_key

# Get the OpenAI API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Set the OpenAI API key if it exists
if openai_api_key:
    openai.api_key = openai_api_key
else:
    # Handle the case where the API key is not set
    raise ValueError("OpenAI API key is not set.")


# # Helper function to process PDFs using pdfplumber
# def get_pdf_docs(folder_path):
#     folder = pathlib.Path(folder_path)
#     print("Iterating through PDF files")

#     # Iterate over only .pdf files in the folder (including subdirectories)
#     for pdf_file in folder.glob("**/*.pdf"):
#         print(pdf_file)
#         with pdfplumber.open(pdf_file) as pdf:
#             text = ""
#             for page in pdf.pages:
#                 text += page.extract_text()
#             yield Document(page_content=text, metadata={"source": str(pdf_file.relative_to(folder))})

def get_pdf_docs(folder_path):
    folder = pathlib.Path(folder_path)
    print("Iterating through PDF files")

    # Iterate over only .pdf files in the folder (including subdirectories)
    for pdf_file in folder.glob("**/*.pdf"):
        print(f"Processing PDF file: {pdf_file}")

        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()
            print(f"Extracted text from {pdf_file}:\n{text}")

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

    # Load a QA chain
    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vector_db.as_retriever())
    response_object = qa.run(input_text)

    # Extract the answer from the response object
    query_response = response_object.answer

    # # Example response object
    # response = {
    #     "answer": query_response,  # Store the response text
    #     "metadata": {
    #         "source": "source_name"  # Add necessary metadata
    #     }
    # }

    # Example response object
    response = {
        "role": "bot",  # Add the role to identify the sender
        "content": query_response  # Store the response text
    }

    # Update the session_state with the response message
    session_state['messages'].append(response)

    return response

# Initialise session state variables
session_state = {
    'generated': [],
    'past': [],
    'messages': [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
}

# Define the home route
# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         user_input = request.form.get('user_input')
#         if user_input:
#             query_response = generate_response(user_input)
#             session_state['past'].append(user_input)
#             session_state['generated'].append(query_response)
#     return render_template('index.html', generated=session_state['generated'], past=session_state['past'])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            query_response = generate_response(user_input)
            session_state['past'].append({"role": "user", "content": user_input})
            session_state['generated'].append({"role": "bot", "content": query_response})
    # return render_template('index.html', messages=session_state['messages'])
    return render_template('index.html', generated=session_state['generated'], past=session_state['past'])

# Run the Flask app
if __name__ == '__main__':

    # generate_chroma_db()  ######### call chroma db function from the script.

    # Initialise session state variables
    session_state = {
        'generated': [],
        'past': [],
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    }
    app.run(debug=True)
