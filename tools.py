from imports import *
from utils import *
from prompts import *
import globals_
from beam_tool import Beam_answerer
load_dotenv()




client = PathwayVectorClient(host=os.getenv("PATHWAY_HOST"), port=os.getenv("PATHWAY_PORT"))
retriever = client.as_retriever(search_kwargs={'k' : 20})

def get_key(x):
    return x.page_content[:36]

def get_path(x):
    return x.metadata['path']

def get_content(x):
    s = x.page_content
    return s.replace("----------------CHUNK ENDS HERE ------------------", "")
def merge_two(node1, node2):
    text2 = get_content(node2)
    node1.page_content = node1.page_content + " " + text2
    node1.metadata['merged_keys'].extend(node2.metadata['merged_keys'])
    node1.metadata['merged_keys_metadata'].extend(node2.metadata['merged_keys_metadata'])
    
    return node1


def automerger(nodes, threshold=0.6): 
    merged_nodes = nodes[:]
    
    merged = True  
    while merged:
        merged = False
        n = len(merged_nodes)
        for i in range(n):
            thres = threshold
            if len(get_content(merged_nodes[i])) < 200:
                thres = 0.2
            for j in range(i + 1, n):
                sim = calculate_cosine_similarity(
                    get_content(merged_nodes[i]),
                    get_content(merged_nodes[j])
                )
                if sim > thres:
                    merged_nodes[i] = merge_two(merged_nodes[i], merged_nodes[j])
                   
                    merged_nodes.pop(j)
                    merged = True
                    break
            if merged:
                break
    
    return merged_nodes



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
def calculate_cosine_similarity(string1, string2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([string1, string2])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity[0][0]

def MERGE(docs):
    for x in docs:
        x.metadata['merged_keys'] = [x.page_content[:36]]
        x.metadata['merged_keys_metadata'] = [x.metadata]

    docset = defaultdict(list)

    for doc in docs:
        path = get_path(doc)
        docset[path].append(doc)


    final_docs = []
    for path in docset:
        nodes = docset[path]
        merged_nodes = automerger(nodes)
        final_docs.extend(merged_nodes)

    return final_docs

class RetrieverInput(BaseModel):
    """Input to the retriever."""

    query: str = Field(description="query to look up in retriever")



def func(query: str, client: PathwayVectorClient, bad_docs: set, document_prompt: BasePromptTemplate, document_separator: str):
        """Function to retrieve documents."""
        if globals_.counter_>globals_.THRESHOLD:
            return Beam_answerer(query, client)
        num_docs = len(bad_docs)
        # retrieved_docs = client.as_retriever(search_kwargs={"k": num_docs+20}).get_relevant_documents(query)
        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=client.as_retriever(search_kwargs={"k": num_docs+20})
        )
        reranked_docs = compression_retriever.get_relevant_documents(query)
        good_docs = []
        for i in reranked_docs:
            if get_key(i) not in bad_docs:
                good_docs.append(i)
        good_docs=MERGE(good_docs)
        return document_separator.join(
        format_document(doc, document_prompt) for doc in good_docs
    )

async def afunc(query: str, client: PathwayVectorClient, bad_docs: set, document_prompt: BasePromptTemplate, document_separator: str):
        """Function to retrieve documents."""
        num_docs = len(bad_docs)
        # retrieved_docs = client.as_retriever(search_kwargs={"k": num_docs+20}).aget_relevant_documents(query)
        if globals_.counter_>globals_.THRESHOLD:
            return Beam_answerer(query, client)    
        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=client.as_retriever(search_kwargs={"k": num_docs+20})
        )
        reranked_docs = compression_retriever.aget_relevant_documents(query)

        good_docs = []
        for i in reranked_docs:
            if get_key(i) not in bad_docs:
                good_docs.append(i)
        good_docs=MERGE(good_docs)
        return document_separator.join(
        aformat_document(doc, document_prompt) for doc in good_docs
    )



def master_tool(
    bad_docs: set, 
    client: PathwayVectorClient,
    name: str,
    description: str,  # You can add other parameters as needed
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = "\n\n",
)-> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        bad_docs: Set of bad documents to exclude from retrieval.
        vectorstore: The vectorstore to use for retrieval.
        name: The name of the tool.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.
        document_prompt: The prompt to use for the document. Defaults to None.
        document_separator: The separator to use between documents. Defaults to "\n\n".

    Returns:
        Tool class to pass to an agent.
    """
    # document_prompt = document_prompt or PromptTemplate.from_template("{page_content}")
    document_prompt = PromptTemplate.from_template("""
    Document Summary:
    -----------------
    Source: {source}
    Page Number: {page_number}

    Page Content:
    {page_content}

    -----------------
    """)

    func_ = partial(
        func,
        client = client,
        bad_docs = bad_docs,
        document_prompt=document_prompt,
        document_separator=document_separator,
    )
    afunc_ = partial(
        afunc,
        client = client,
        bad_docs = bad_docs,
        document_prompt=document_prompt,
        document_separator=document_separator,
    )
        
    tool = Tool(
        name=name,
        description=description,
        func=func_,
        coroutine=afunc_,
        args_schema=RetrieverInput,
    )


    return tool

Master_tool = master_tool(globals_.bad_docs, 
                          client,
                          name="1004_master_tool",
                          description="1004_master_tool(query: str) -> str:\n"
    " - Log all bad retrieved documents.\n"
    " - Supervise the retrieval process and find relevant clauses, definitions, and summaries related to contract law, ensuring the information is concise and pertinent.\n",
      document_separator="\n\n")
