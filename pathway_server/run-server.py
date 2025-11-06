
import os
TESSDATA_PREFIX = "usr/share/tesseract-ocr/5/tessdata" #Example: /usr/share/tesseract-ocr/4.00/tessdata
os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX 

import openai
import atexit
import signal
import sys


from pathway.xpacks.llm.vector_store import VectorStoreServer, VectorStoreClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import TokenTextSplitter
from pathway.xpacks.llm import parsers, embedders
import pathway as pw
import time

from pathway.udfs import DiskCache, ExponentialBackoffRetryStrategy
from pathway.xpacks.llm import embedders, llms, parsers, prompts
from pathway.xpacks.llm.question_answering import BaseRAGQuestionAnswerer
from server import VectorStoreServer
from pathway.udfs import DiskCache
from llama_index.retrievers.pathway import PathwayRetriever

from io import StringIO
openai.api_key = os.environ.get('OPENAI_API_KEY')
output_buffer = StringIO()

def cleanup(server_thread):
    """Cleanup function to handle graceful shutdown"""
    if server_thread and server_thread.is_alive():
        print("Shutting down server...")
        sys.exit(0)


from custom_parser import CustomParse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import SecretStr


def main():
    data_sources = []
    data_sources.append(
        pw.io.fs.read(
            "data",
            format="binary",
            mode="streaming",
            with_metadata=True,
        )  # This creates a `pathway` connector that tracks
        # all the files in the ./data directory
    )
    
    parser = CustomParse()
    
    # embeddings = VoyageAIEmbeddings(
    #     voyage_api_key="INSERT VOYAGEAI KEY HERE", model="voyage-3"
    # )
    embeddings = OpenAIEmbeddings(api_key=SecretStr(os.environ["OPENAI_API_KEY"]))

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=50
    )
    embeddings_model = OpenAIEmbeddings(api_key=SecretStr(os.environ["OPENAI_API_KEY"]))

    vector_server = VectorStoreServer.from_langchain_components(
        *data_sources,
        embedder=embeddings_model,
        splitter=text_splitter,
        parser = parser
    )
    
    server_thread = vector_server.run_server(host="127.0.0.1", port=8745, threaded=True, with_cache=True)
    
    # Register cleanup handlers
    atexit.register(cleanup, server_thread)
    signal.signal(signal.SIGINT, lambda s, f: cleanup(server_thread))
    signal.signal(signal.SIGTERM, lambda s, f: cleanup(server_thread))

    try:
        # Keep main thread alive
        server_thread.join()
    except KeyboardInterrupt:
        cleanup(server_thread)
        
if __name__ == "__main__":
    main()
    
    
    
    