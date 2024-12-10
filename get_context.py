# Combining 'preprocess' and 'retriever' to extract the relevant context of a query in one step

import preprocess
import retriever

def getContextDocs(urls, query):
    def uploadDocs(urls):
        for url in urls:
            preprocess.download_file(url)
        return preprocess.ingestDocs()
    document_store = uploadDocs(urls)
    return (retriever.retrieve(document_store, query))['documents']

def getContext(urls, query):
    contexts = getContextDocs(urls, query)
    return contexts[0].content
