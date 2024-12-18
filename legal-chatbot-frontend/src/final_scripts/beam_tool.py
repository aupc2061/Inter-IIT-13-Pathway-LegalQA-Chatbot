from beam_retriever import BeamRetriever,BEAM_HEAD
from langchain_community.vectorstores import PathwayVectorClient

# client = PathwayVectorClient(host="127.0.0.1", port=8745)
def Beam_answerer(question, client):

  try:
    head = BEAM_HEAD()
    head.load('fine_tuned.pkl')
    Beam = BeamRetriever(head,B=3, K=3)
    List = client.similarity_search_with_relevance_scores(question, k = 20)
    D = {}
    i = 0
    for x in List:
      D['node' + str(i)] = x[0].page_content[36:].strip()
      i+=1
    
    beam_nodes = Beam.retrieve(question, D)
    s = ''
    for nodes in beam_nodes:
      s += D[nodes]
    return s
  except:
    return None

# Beam_answerer('sterling winters', client)