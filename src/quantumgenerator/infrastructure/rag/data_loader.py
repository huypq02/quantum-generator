from langchain_community.document_loaders import PyPDFLoader


def load_data(file_path: str):
    """
    Docstring for load_data
    """
    loader = PyPDFLoader(file_path)
    return loader.load()
