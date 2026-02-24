from typing import Any, Sequence

from langchain_text_splitters import TokenTextSplitter


def chunking(
        encoding_name: str,
        chunk_size: int,
        chunk_overlap: int,
    doc_list: Sequence[Any],
):
    """
    Convert text documents to tokens using a text splitter.
    
    :param encoding_name: Name of the token encoder to use.
    :type encoding_name: str
    :param chunk_size: Maximum size of each chunk in tokens.
    :type chunk_size: int
    :param chunk_overlap: Number of overlapping tokens between chunks.
    :type chunk_overlap: int
    :param doc_list: List of loaded documents to chunk.
    :type doc_list: Sequence[Any]
    :return: Tokenized and chunked documents.
    :rtype: list
    """
    token_splitter = TokenTextSplitter(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return token_splitter.split_documents(doc_list)
