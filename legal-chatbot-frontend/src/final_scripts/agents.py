from imports import *

import globals_
from anthropic_functions import ChatAnthropic
from utils import *
from tools import *
from prompts import *
from joiner import *


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """


    messages = state["messages"]
    question = messages[0].content
    # print(counter_)
    if globals_.counter_ <= 3:
        context = messages[-3]
    else:
        context = messages[-4]


    # GRADE DOCUMENTS
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = llm_with_fallback.call()

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt

    if isinstance(llm_with_fallback.call(), ChatOpenAI):
        prompt = ChatPromptTemplate([
                ("system", """You are an AI-powered relevance assessment system designed to evaluate the relevance of retrieved documents to user questions. Your task is to carefully analyze the provided document and question, considering various aspects of relevance, and then provide a binary relevance score.

                Here is the retrieved document:
                <context>
                {{context}}
                </context>

                Here is the user question:
                <question>
                {{question}}
                </question>

                Instructions:
                1.⁠ ⁠Carefully read both the context and the question.
                2.⁠ ⁠In your analysis, consider the following aspects of relevance:
                - Presence of keywords from the question in the context
                - Semantic similarity between the question and the context
                - Potential indirect or implicit relevance
                - Any information in the context that could contribute to answering the question, even partially
                3.⁠ ⁠Conduct your analysis inside <relevance_analysis> tags, following these steps:
                a. Extract and quote relevant parts of the context that relate to the question.
                b. List key concepts from the question and check if they appear in the context.
                c. Consider potential indirect relevance or implications.
                d. Weigh arguments for and against relevance.
                4.⁠ ⁠After your analysis, provide a binary score: 'yes' if the document is relevant, or 'no' if it is irrelevant.
                5.⁠ ⁠Remember, if you find any kind of similarity or potentially useful information, no matter how small, lean towards grading it as relevant.\n""" ), 
                ("human", "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.")])
    elif isinstance(llm_with_fallback.call(), ChatAnthropic):
       prompt = PromptTemplate(
                template="""You are an AI-powered relevance assessment system designed to evaluate the relevance of retrieved documents to user questions. Your task is to carefully analyze the provided document and question, considering various aspects of relevance, and then provide a binary relevance score.

                                Here is the retrieved document:
                                <context>
                                {{context}}
                                </context>

                                Here is the user question:
                                <question>
                                {{question}}
                                </question>

                                Instructions:
                                1.⁠ ⁠Carefully read both the context and the question.
                                2.⁠ ⁠In your analysis, consider the following aspects of relevance:
                                - Presence of keywords from the question in the context
                                - Semantic similarity between the question and the context
                                - Potential indirect or implicit relevance
                                - Any information in the context that could contribute to answering the question, even partially
                                3.⁠ ⁠Conduct your analysis inside <relevance_analysis> tags, following these steps:
                                a. Extract and quote relevant parts of the context that relate to the question.
                                b. List key concepts from the question and check if they appear in the context.
                                c. Consider potential indirect relevance or implications.
                                d. Weigh arguments for and against relevance.
                                4.⁠ ⁠After your analysis, provide a binary score: 'yes' if the document is relevant, or 'no' if it is irrelevant.
                                5.⁠ ⁠Remember, if you find any kind of similarity or potentially useful information, no matter how small, lean towards grading it as relevant.\n""",
input_variables=['context', 'question'])

    # Chain
    def dum(state) -> dict:
        return state
    chain = dum | prompt | llm_with_tool | debug_output
    
    ret_docs=[]
    
    regex2 = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    doc_keys_temp = []
# Find all matches
    for i in messages[::-1]:
        if isinstance(i, FunctionMessage) or isinstance(i, ToolMessage):
            if i.content == "join":
                continue
            chunks = i.content.split('\n----------------CHUNK ENDS HERE ------------------\n')

            items=[]
            chunks = chunks[:len(chunks)-1]
            for chunk in chunks:
                item=str(re.findall(r'Page Content:([\s\S]+)', str(chunk))[0])
                items.append(item)
            item_keys = re.findall(regex2, i.content)
            for j in items:
                ret_docs.append(j)
            for j in item_keys:
                doc_keys_temp.append(j)

    doc_to_keys={}
    for i in range(len(ret_docs)):
        doc_to_keys[ret_docs[i]]=doc_keys_temp[i]
    good_docs = []
    for docs in ret_docs:
        try:
            scored_result = chain.invoke({"question": question, "context": docs})
        except ValueError:
            print("Couldn't get the score")
        score = scored_result.binary_score
        if score == "yes":
            good_docs.append(docs)
            continue
        else:
            globals_.bad_docs.add(doc_to_keys[docs])
    

    msg = [
        HumanMessage(
            content=f""" \n
    Look at the input and try to reason about the underlying semantic intent / meaning. \n
    Here is the initial question:
    \n ------- \n
    {question}
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model = llm_with_fallback.call()
    try:
        response = model.invoke(msg)
    except ValueError:
        print("Model call failed")
    
    if len(good_docs) == 0:
        last_response = "No Relevant documents retrieved in the last plan. Please try again."
    else:
        last_response = f"Relevant documents retrieved in the last plan are as follows: {good_docs}"
    new_response = f"\n REWRITTEN QUERY: {response.content}"
    return {"messages": messages + [HumanMessage(content=last_response)] + [SystemMessage(content=new_response)]}


def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """

    messages = state["messages"]

    question = messages[0].content
    if globals_.counter_ <= 3:
        last_message = messages[-3]
    else:
        last_message = messages[-4]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = llm_with_fallback.call()

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | debug_output| StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


#cachePath = os.getcwd()
cachePath = "cache1.txt"


# Function to read data from the cache.txt file
def get_cache_data():
    with open(cachePath, "r") as file:
        data = file.read().strip()
    return data


def write_cache_data(feedback, flag):
    with open(cachePath, "w") as file:
        file.write(f"flag = {flag}")

# Function to wait for feedback input from the frontend
def get_feedback_from_frontend():
    with open(cachePath, "w") as file:
        file.write("flag = True")

    with open("test.txt", "w") as file:
        file.write("test")
    i = 0
    while True:
        i+=1
        with open("test.txt", "w") as file:
            file.write(f"entered true loop + {i}")
        cache_data = get_cache_data()
        
        # Check if the flag is True, indicating that the frontend is ready for input
        if "flag = True" in cache_data:
            print("Waiting for feedback from the frontend...")
            time.sleep(0.1) 
        else:
            # Once the flag is False, the input has been provided
            feedback_start = cache_data.find("feedback = ") + len("feedback = ")
            feedback = cache_data[feedback_start:].strip()
            
            print(f"Feedback received: {feedback}")
            
            # Set the flag back to False after receiving feedback
            write_cache_data(feedback, "False")
            return feedback
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def human_input_node(state):

    # input_ = "thhese documents are not good, pls replan you dumb fuck"
    input_ = get_feedback_from_frontend()
    #input("Enter your feedback")
    messages = state["messages"]
    if input_ == "":
        return {"messages" : messages}

    
    messages.append(SystemMessage(
                    content=input_
                ))

    return {"messages": messages}

