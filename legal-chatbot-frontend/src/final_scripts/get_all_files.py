from langchain_community.vectorstores import PathwayVectorClient
import json
from globals_ import PATHWAY_SERVER_URL
client = PathwayVectorClient(url = PATHWAY_SERVER_URL)


def get_all_files():
    l = client.get_input_files()
    return json.dumps(l)
