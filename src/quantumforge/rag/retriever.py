from langchain_text_splitters import TokenTextSpitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chorma.vectorstores import Chorma
from langchain_community.document_loaders import PyPDFLoader
import os

def load_retriever(file: str = "Intro-to-AI-notes.pdf"):
    try:
        directory_path='/'.join("data", file) # TODO: should be set directory in config
        if os.path.isdir(directory_path) == False:
            print("File not found")
            return
        
        # Load PDF file
        load_pdf = PyPDFLoader(directory_path)
        doc_list = load_pdf.load()
        
        # Convert text to tokens
        token_splitter = TokenTextSpitter(encoding_name="cl100k_base",
                                          chunk_size=200,
                                          chunk_overlap=40)
        doc_list_token_splitter = token_splitter.split_documents(doc_list)

        # Using HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # TODO: create a vector store and after that finding the most relevant to the query
        
    except Exception as e:
        print(f"Error while processing retriever: {e}")
        raise