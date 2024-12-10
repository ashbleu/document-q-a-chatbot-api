# Preprocessing and ingesting input documents into the Qdrant vector database

import requests
import os

def urlToTxt(url: str) -> str:
    response = requests.get(url)
    return(response.text)

def download_file(url):
    local_filename = url.split('/')[-1]
    os.makedirs("./docs", exist_ok=True)
    path = os.path.join("./docs", local_filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)
    return path

import shutil

def clearDocs():    
    try:
        folder_path = './docs'
        shutil.rmtree(folder_path)
        print('Folder deleted')
    except:
        print('Folder not deleted')

from haystack.components.writers import DocumentWriter
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack import Pipeline

def embed(query: str):
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    text_embedder.warm_up()
    return (text_embedder.run(query)['embedding'])

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

def ingestDocs() -> QdrantDocumentStore:
    document_store = QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
            embedding_dim=384
        )
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_file_converter = TextFileToDocument()
    markdown_converter = MarkdownToDocument()
    pdf_converter = PyPDFToDocument()
    document_joiner = DocumentJoiner()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
    document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    document_writer = DocumentWriter(document_store)

    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")

    preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    preprocessing_pipeline.connect("text_file_converter", "document_joiner")
    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("markdown_converter", "document_joiner")
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    from pathlib import Path

    preprocessing_pipeline.run({"file_type_router": {"sources": list(Path("./docs").glob("**/*"))}})

    return document_store
