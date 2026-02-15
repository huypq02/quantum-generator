from typing import Any, Sequence

from langchain_text_splitters import TokenTextSplitter


def chunking(
        encoding_name: str,
        chunk_size: int,
        chunk_overlap: int,
    doc_list: Sequence[Any],
):
    """
    Convert text to tokens.

    :param encoding_name: Encoder.
    :param chunk_size: Chunking size.
    :param chunk_overlap: Chunking overlap.
    :param doc_list: Loaded documents.
    :return: tokenized documents
    """
    token_splitter = TokenTextSplitter(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    return token_splitter.split_documents(doc_list)
