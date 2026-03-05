"""
ZX Bank AI — TF-IDF Keyword Extractor.

Extracts top keywords from each document chunk using TF-IDF.
Keywords are stored in chunk metadata and used to boost BM25
retrieval relevance.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from app.retrieval.document_processor import DocumentChunk
from app.utils.logger import get_logger

logger = get_logger(__name__, component="tfidf")


def extract_keywords(
    chunks: list[DocumentChunk],
    top_n: int = 10,
) -> list[DocumentChunk]:
    """
    Enrich each chunk's metadata with TF-IDF extracted keywords.

    Args:
        chunks: List of document chunks to process.
        top_n: Number of top keywords per chunk.

    Returns:
        The same list of chunks with ``metadata["keywords"]`` populated.
    """
    if not chunks:
        return chunks

    corpus = [chunk.content for chunk in chunks]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.85,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    for idx, chunk in enumerate(chunks):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        keywords = [
            feature_names[i]
            for i in top_indices
            if row[i] > 0
        ]
        chunk.metadata["keywords"] = keywords

    logger.info(
        "tfidf_extraction_complete",
        total_chunks=len(chunks),
        top_n=top_n,
    )
    return chunks
