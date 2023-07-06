import os
import pathlib
import pdfplumber
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import PythonCodeTextSplitter
from langchain.schema import Document

# Code for extracting PDF documents
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

# Code for creating source chunks
# Use the Python code text splitter from Langchain to create chunks
def get_source_chunks(repo_path, pdf_folder_path): 
    source_chunks = []
    print("Creating source chunks")

    # Create a PythonCodeTextSplitter object for splitting the code
    splitter = PythonCodeTextSplitter(chunk_size=1024, chunk_overlap=30)

    for pdf in get_pdf_docs(pdf_folder_path):
        for chunk in splitter.split_text(pdf.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=pdf.metadata))
    return source_chunks

def generate_chroma_db():
    # Define the path of the repository and Chroma DB
    REPO_PATH = os.path.abspath(os.path.dirname(__file__))
    CHROMA_DB_PATH = os.path.join(REPO_PATH, "chromaDB")

    # Check if Chroma DB exists
    if not os.path.exists(CHROMA_DB_PATH):
        # Create a new Chroma DB
        print(f"Creating Chroma DB at {CHROMA_DB_PATH}...")
        source_chunks = get_source_chunks(REPO_PATH, os.path.join(REPO_PATH, "sample_policies"))
        vector_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)
        vector_db.persist()
    else:
        print(f"Chroma DB already exists at {CHROMA_DB_PATH}")

if __name__ == "__main__":
    generate_chroma_db()
