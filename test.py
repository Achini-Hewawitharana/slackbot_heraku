import os
import subprocess
import sys
import pathlib
from flask import Flask, render_template, request
# from generate_chroma_db import generate_chroma_db

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

openai_api_key = os.environ.get("OPENAI_API_KEY")

# Set the OpenAI API key if it exists
if openai_api_key:
    openai.api_key = openai_api_key
else:
    # Handle the case where the API key is not set
    raise ValueError("OpenAI API key is not set.")

def get_pdf_docs(folder_path):
    print("get_pdf_docs function is called")
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
    splitter = PythonCodeTextSplitter(chunk_size=1024, chunk_overlap=30)

    for pdf in get_pdf_docs(pdf_folder_path):
        for chunk in splitter.split_text(pdf.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=pdf.metadata))
    return source_chunks

def generate_response(input_text):
    REPO_PATH = os.path.abspath(os.path.dirname(__file__))
    CHROMA_DB_PATH = os.path.join(REPO_PATH, "chromaDB")

    vector_db = None

    if not os.path.exists(CHROMA_DB_PATH):
        source_chunks = get_source_chunks(REPO_PATH, os.path.join(REPO_PATH, "sample_policies"))
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        vector_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        vector_db.persist()
    else:
        print(f'Loading Chroma DB from {CHROMA_DB_PATH}...')
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

    # Load a QA chain
    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vector_db.as_retriever())
    query_response = qa.run(input_text)

    response_text = query_response[0]["content"] if query_response else "I'm sorry, I don't have an answer."

    response = {
        "role": "bot",
        "content": response_text
    }
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

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         user_input = request.form.get('user_input')
#         if user_input:
#             query_response = generate_response(user_input)
#             session_state['past'].append({"role": "user", "content": user_input})
#             session_state['generated'].append({"role": "user", "content": user_input})
#             session_state['generated'].append({"role": "bot", "content": query_response})
#     else:
#         if len(session_state['generated']) == 0 and len(session_state['past']) == 0:
#             # Greet the user and introduce the chatbot
#             session_state['generated'].append({"role": "bot", "content": "Hi there! I'm your friendly chatbot. How can I assist you today?"})

#     return render_template('index.html', generated=session_state['generated'], past=session_state['past'])

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            session_state['past'].append({"role": "user", "content": user_input})
            query_response = generate_response(user_input)
            session_state['generated'].append({"role": "user", "content": user_input})
            session_state['generated'].append({"role": "bot", "content": query_response["content"]})
    else:
        if len(session_state['generated']) == 0 and len(session_state['past']) == 0:
            # Greet the user and introduce the chatbot
            session_state['generated'].append({"role": "bot", "content": "Hi there! How can I assist you today?"})

    return render_template('index2.html', generated=session_state['generated'], past=session_state['past'])


# Run the Flask app
if __name__ == '__main__':
    # Initialise session state variables
    session_state = {
        'generated': [],
        'past': [],
        'messages': [
            {"role": "system", "content": "You are a helpful assistant. First greet the user. Give \
             answers in numbered list format."}
        ]
    }
    app.run(debug=True)
