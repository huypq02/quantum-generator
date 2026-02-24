from langchain_community.document_loaders import PyPDFLoader


def load_data(file_path: str):
    """
    Load PDF documents from the specified file path.
    
    :param file_path: Path to the PDF file.
    :type file_path: str
    :return: Loaded document objects.
    :rtype: list
    """
    loader = PyPDFLoader(file_path)
    return loader.load()
