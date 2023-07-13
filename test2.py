import os
import pathlib
import pdfplumber
import openai
from flask import Flask, render_template, request
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain

# Define the Flask app
app = Flask(__name__)

# Get the OpenAI API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")

# Set the OpenAI API key if it exists
if openai_api_key:
    openai.api_key = openai_api_key
    print("OpenAI API key is set")
else:
    # Handle the case where the API key is not set
    raise ValueError("OpenAI API key is not set.")

REPO_PATH = os.path.abspath(os.path.dirname(__file__))
CHROMA_DB_PATH = os.path.join(REPO_PATH, "chromaDB")

def generate_response(input_text):
    print("generate_response function is called")

    vector_db = None
    existing_filenames = []

    # Check if Chroma DB exists
    if not os.path.exists(CHROMA_DB_PATH):
        # Create a new Chroma DB
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        source_chunks = get_source_chunks(REPO_PATH, os.path.join(REPO_PATH, "sample_policies"))
        vector_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        vector_db.persist()
        existing_filenames = [doc.metadata.get("source") for doc in source_chunks]
    else:
        # Load an existing Chroma DB
        print(f'Loading Chroma DB from {CHROMA_DB_PATH}...')
        vector_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings())

        # Check if there are new PDF documents
        source_chunks = list(get_source_chunks(REPO_PATH, os.path.join(REPO_PATH, "sample_policies")))
        new_documents = []

        for doc in source_chunks:
            filename = doc.metadata.get("source")
            if filename not in existing_filenames:
                new_documents.append(doc)
                existing_filenames.append(filename)

        if new_documents:
            print(f"Found {len(new_documents)} new document(s). Updating Chroma DB...")

            retriever = vector_db.as_retriever()

            for doc in new_documents:
                retriever.add_document(doc)

            vector_db = Chroma.from_retriever(retriever, persist_directory=CHROMA_DB_PATH)
            vector_db.persist()

    # Load a QA chain
    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vector_db.as_retriever())
    query_response = qa.run(input_text)

    response = {
        "role": "bot",
        "content": query_response
    }

    return response

# Function to get source code chunks from PDF documents
def get_source_chunks(repo_path, pdf_folder_path):
    source_chunks = []

    # Create a PythonCodeTextSplitter object for splitting the code
    splitter = PythonCodeTextSplitter(chunk_size=1024, chunk_overlap=30)

    for pdf in get_pdf_docs(pdf_folder_path):
        for chunk in splitter.split_text(pdf.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=pdf.metadata))
    return source_chunks

# Function to get PDF documents from a folder
def get_pdf_docs(folder_path):
    folder = pathlib.Path(folder_path)

    # Iterate over only .pdf files in the folder (including subdirectories)
    for pdf_file in folder.glob("**/*.pdf"):
        print(f"Processing PDF file: {pdf_file}")

        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        print(f"Extracted text from {pdf_file}:\n{text}")

        yield Document(page_content=text, metadata={"source": str(pdf_file.relative_to(folder))})

# Initialise session state variable
session_state = {
    'messages': [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input:
            query_response = generate_response(user_input)
            session_state['messages'].append({"role": "user", "content": user_input})
            session_state['messages'].append({"role": "bot", "content": query_response['content']})
    else:
        query_response = generate_response("Hello")
        session_state['messages'].append({"role": "user", "content": ""})
        session_state['messages'].append({"role": "bot", "content": query_response['content']})

    return render_template('index3_2.html', messages=session_state['messages'])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
