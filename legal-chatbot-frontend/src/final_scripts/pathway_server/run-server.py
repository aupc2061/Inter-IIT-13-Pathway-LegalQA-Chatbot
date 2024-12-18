
import os
TESSDATA_PREFIX = "INSERT TESSDATA PREFIX HERE" #Example: /usr/share/tesseract-ocr/4.00/tessdata
os.environ["TESSDATA_PREFIX"] = TESSDATA_PREFIX 

import openai


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
openai.api_key = "INSERT OPENAI_KEY HERE"
os.environ['OPENAI_API_KEY'] = openai.api_key
output_buffer = StringIO()


data_sources = []
data_sources.append(
    pw.io.fs.read(
        "INSERT PATH TO DATA DIRECTORY HERE",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )  # This creates a `pathway` connector that tracks
    # all the files in the ./data directory
)

from custom_parser import CustomParse
parser = CustomParse()
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

embeddings = VoyageAIEmbeddings(
    voyage_api_key="INSERT VOYAGEAI KEY HERE", model="voyage-3"
)

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=50
)
embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

vector_server = VectorStoreServer.from_langchain_components(
    *data_sources,
    embedder=embeddings,
    splitter=text_splitter,
    parser = parser
)

vector_server.run_server(host="127.0.0.1", port=8745, threaded=True, with_cache=True)