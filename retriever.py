# Retrieving the most relevant context by performing a search on the vector database

from haystack_integrations.components.retrievers.qdrant import retriever as qr

from preprocess import embed

def retrieve(document_store, query):
    retriever = qr.QdrantEmbeddingRetriever(document_store)
    return retriever.run(embed(query))
