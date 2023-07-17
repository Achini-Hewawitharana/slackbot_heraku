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
# openai_api_key = os.environ.get("OPENAI_API_KEY")

openai_api_key = "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"

# Set the OpenAI API key if it exists
if openai_api_key:
    openai.api_key = openai_api_key
else:
    # Handle the case where the API key is not set
    raise ValueError("OpenAI API key is not set.")

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

# Function to get source code chunks from PDF documents
def get_source_chunks(repo_path, pdf_folder_path): 
    source_chunks = []

    # Create a PythonCodeTextSplitter object for splitting the code
    splitter = PythonCodeTextSplitter(chunk_size=1024, chunk_overlap=30)

    for pdf in get_pdf_docs(pdf_folder_path):
        for chunk in splitter.split_text(pdf.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=pdf.metadata))
    return source_chunks


# Function to generate response from user input
def generate_response(input_text):
    # Define the path of the repository and Chroma DB
    # to get the absolute path of the current script file
    REPO_PATH = os.path.abspath(os.path.dirname(__file__))
    CHROMA_DB_PATH = os.path.join(REPO_PATH, "chromaDB")

    vector_db = None

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
    qa_chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vector_db.as_retriever())
    query_response = qa.run(input_text)
    print("query_response =", query_response)

    # # Generate a response from ChatGPT
    # chat_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": input_text}
    #     ]
    # )
    # chat_response_text = chat_response.choices[0].message.content

    # # Combine the response from the QA chain and ChatGPT
    # combined_response = query_response + "\n" + chat_response_text
    combined_response = query_response

    # Example response object
    response = {
        "role": "bot",  # Add the role to identify the sender
        "content": combined_response  # Store the combined response text
    }

    return response

# Initialise session state variable
# session_state = {
#     'messages': [
#         {"role": "system", "content": "You are a helpful assistant."}
#     ]
# }

session_state = {
    'messages': [
        {"role": "system", "content": "Welcome! How can I assist you today?"}
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
    elif len(session_state['messages']) == 0:
        # Only display the initial message if there are no existing messages
        query_response = generate_response("Hello")
        session_state['messages'].append({"role": "user", "content": ""})
        session_state['messages'].append({"role": "bot", "content": query_response['content']})

    return render_template('index4_1.html', messages=session_state['messages'])


# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         user_input = request.form.get('user_input')
#         if user_input:
#             query_response = generate_response(user_input)
#             session_state['messages'].append({"role": "user", "content": user_input})
#             session_state['messages'].append({"role": "bot", "content": query_response['content']})
#     else:
#         # Only display the initial message on the first GET request
#         if len(session_state['messages']) == 1:
#             query_response = generate_response("Hello")
#             session_state['messages'].append({"role": "user", "content": ""})
#             session_state['messages'].append({"role": "bot", "content": query_response['content']})

#     return render_template('index4_1.html', messages=session_state['messages'])


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
