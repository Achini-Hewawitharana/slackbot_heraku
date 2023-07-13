import os
import pathlib
import pdfplumber
from flask import Flask, render_template, request
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

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


def generate_response(input_text):
    # Check if the user is asking about company policies at DxDy
    if "DxDy company policies" in input_text:
        # Use the policies from the 'sample_policies' folder and chromadb
        response = get_policy_response(input_text)
    else:
        # Use the model's knowledge base for common knowledge policies
        response = get_common_knowledge_response(input_text)

    # Example response object
    response_object = {
        "role": "bot",  # Add the role to identify the sender
        "content": response  # Store the response text
    }

    # Update the session_state with the response message
    session_state['messages'].append(response_object)

    return response


def get_policy_response(input_text):
    # Define the path of the repository and Chroma DB
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

    # Load a RetrievalQA chain
    qa_chain = RetrievalQA(retriever=vector_db.as_retriever())
    query_response = qa_chain.run(input_text)

    if query_response:
        return query_response[0].answer_text
    else:
        return "There's no proper answer."


def get_common_knowledge_response(input_text):
    # Use the GPT-3.5 Turbo model to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",  # Use the GPT-3.5 Turbo model
        prompt=input_text,
        max_tokens=100,  # Adjust as needed
        n=1,  # Number of completions to generate
        stop=None,  # Stop generating completions at this sequence
        temperature=0.7  # Adjust as needed
    )

    if response.choices:
        return response.choices[0].text.strip()
    else:
        return "There's no proper answer."


# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         user_input = request.form.get('user_input')
#         if user_input:
#             query_response = generate_response(user_input)
#             session_state['past'].append({"role": "user", "content": user_input})
#             session_state['generated'].append({"role": "bot", "content": query_response})
#     else:
#         # Add a default response when the page is loaded initially
#         query_response = generate_response("Hello")
#         session_state['generated'].append({"role": "bot", "content": query_response})

#     return render_template('index2.html', generated=session_state['generated'], past=session_state['past'])

@app.route('/', methods=['GET', 'POST'])
def home():
    print("home route is accessed")
    if request.method == 'POST':
        print("POST is accessed")
        user_input = request.form.get('user_input')
        if user_input:
            query_response = generate_response(user_input)
            session_state['past'].append({"role": "user", "content": user_input})
            session_state['generated'].append({"role": "bot", "content": query_response['content']})
            # Update the session_state after generating the bot's response
            session_state['messages'].append(query_response['content'])  # Update this line
    else:
        print("ELSE is accessed")
        # Add a default response when the page is loaded initially
        query_response = generate_response("Hello")
        session_state['past'].append({"role": "user", "content": ""})
        session_state['generated'].append({"role": "bot", "content": query_response['content']})
        # Update the session_state after generating the default bot's response
        session_state['messages'].append(query_response['content'])  # Update this line
    
    return render_template('index2.html', generated=session_state['generated'], past=session_state['past'])


# Initialise session state variables
session_state = {
    'generated': [],
    'past': [],
    'messages': [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
}

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
