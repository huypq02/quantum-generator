from .factory import EmbeddingModel
from langchain_text_splitters import TokenTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
from pathlib import Path

ROOT_DIR = Path(__file__).parents[3]
DOCS_DIR = os.path.join(ROOT_DIR, "data", "quantum_docs")
VECTORDB_DIR = os.path.join(ROOT_DIR, "data", "vectordb", "chroma")
DEFAULT_FILE_PATH = os.path.join("general", "Intro-to-AI-notes.pdf")
EMBEDDING_TYPE = "minilm-l6"

def load_retriever(file: str = DEFAULT_FILE_PATH):
    try:
        file_path = os.path.join(DOCS_DIR, file)
        if not os.path.isfile(file_path):
            print(f"File not found at path {file_path}")
            raise FileNotFoundError('File not found')
        
        # Load PDF file
        load_pdf = PyPDFLoader(file_path)
        doc_list = load_pdf.load()
        
        # Convert text to tokens
        token_splitter = TokenTextSplitter(encoding_name="cl100k_base",
                                           chunk_size=200,
                                           chunk_overlap=40)
        doc_list_token_splitter = token_splitter.split_documents(doc_list)

        # Using HuggingFace embeddings
        embeddings = EmbeddingModel.create_embeddings(EMBEDDING_TYPE)

        # Check if a directory exists or not then create new
        os.makedirs(VECTORDB_DIR, exist_ok=True)

        # Create a Chroma vector database that keeps all vector in the local directory
        vectorstore = Chroma.from_documents(documents=doc_list_token_splitter,
                                            embedding=embeddings,
                                            persist_directory=VECTORDB_DIR)
        
        # Finding the most relevant document to the query
        retriever = vectorstore.as_retriever(search_type="mmr",     # Using Maximal Marginal Relevance algorithm
                                             search_kwargs={
                                                 'k':1,             # Top 1 most relevant document
                                                 'lambda_mult':0.7, # 70% focus on relevance, 30% on diversity
                                             })
        
        return retriever

    except Exception as e:
        print(f"Error while processing retriever: {e}")
        raise
