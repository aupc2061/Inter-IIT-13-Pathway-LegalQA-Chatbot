from imports import *

from joiner import _parse_joiner_output
from utils import *
from tools import *
from prompts import *
from planner import *
from task_fetching_unit import *
from joiner import *
from agents import *
from HIL import *
from HIL import _parse_feedback_output
from citations import *
import warnings
import globals_
from globals_ import MONGO_URL

warnings.filterwarnings('ignore')



tools=[Master_tool]

planner = create_planner(llm_with_fallback.call(), tools, prompt)

import itertools
import copy

load_dotenv()

@as_runnable
def plan_and_schedule(state):
    messages = state["messages"]
    new_msgs = [messages[0], messages[-1]]
    tasks = planner.stream(messages)
    if(isinstance(llm_with_fallback.call(), ChatOpenAI)):
        final_messages=messages
    elif(isinstance(llm_with_fallback.call(), ChatAnthropic)):
        final_messages=new_msgs
    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except:
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": final_messages,
            "tasks": tasks,
        }
    )
    return {"messages": scheduled_tasks}


runnable = joiner_prompt | llm_with_fallback.call().with_structured_output(JoinOutputs)
joiner = select_recent_messages | runnable | _parse_joiner_output
runnable2 = feedback_prompt | llm_with_fallback.call().with_structured_output(HumanFeedback) 
feedback = select_last_message | runnable2 | _parse_feedback_output

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


graph_builder = StateGraph(State)

# 1.  Define vertices
# We defined plan_and_schedule above already
# Assign each node to a state variable to update
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)
graph_builder.add_node("HIL", feedback)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("generate", generate)
graph_builder.add_node("human_input_node", human_input_node)


## Define edges
graph_builder.add_edge("plan_and_schedule", "join")

def should_continue(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage):
        return "generate"
    return "rewrite"


def feedback_or_not(state):
    if globals_.counter_ > 3:
        return "human_input_node"
    else:
        messages = state["messages"]
        if isinstance(messages[-1], AIMessage):
            return "generate"
        return "rewrite"

graph_builder.add_conditional_edges(
    "join",
    # Next, we pass in the function that will determine which node is called next.
    feedback_or_not
)
graph_builder.add_edge("human_input_node", "HIL")

graph_builder.add_conditional_edges(
    "HIL",
    # Next, we pass in the function that will determine which node is called next.
    should_continue
)

graph_builder.add_edge('rewrite', 'plan_and_schedule')
graph_builder.add_edge('generate', END)
graph_builder.add_edge(START, "plan_and_schedule")
# Set up memory
memory = MemorySaver()
chain = graph_builder.compile()


langfuse_handler = CallbackHandler(

  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),

  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),

  host=os.getenv("LANGFUSE_HOST")

)


# MongoDB connection setup
client = MongoClient(MONGO_URL)
db = client['legal_chatbot']  # Use your database name here
collection = db['script_response']  # Collection where we store responses


def join_txt(input_string):
    docs = []
    # Split the input string by the marker for each document
    chunks = input_string.split("Document Summary:")
    thought = chunks[0]
    chunks = chunks[1:]
    docs.append({
        "thought": thought
    })
    for chunk in chunks:
        if not chunk.strip():
            continue

        source_start = chunk.find("Source:") + len("Source:")
        source_end = chunk.find("Page Number:")
        source = chunk[source_start:source_end].strip()


        page_number_start = chunk.find("Page Number:") + len("Page Number:")
        page_number_end = chunk.find("Page Content:")
        page_number = chunk[page_number_start:page_number_end].strip()


        page_content_start = chunk.find("Page Content:") + len("Page Content:")
        page_content_end = chunk.find("----------------CHUNK ENDS HERE ------------------")
        page_content = chunk[page_content_start:page_content_end].strip()

        if source and page_number and page_content:
            docs.append({
                "source": source,
                "page_number": page_number,
                "page_content": page_content
            })

    return json.dumps(docs, indent=4)

import re
def find_last_text_in_quotes(s):
    matches = re.findall(r"'(.*?)'", s)
    if not matches:
        return ''
    S = matches[-1]
    if '\n' in S:
        return ''
    return S
    

def rewrite_and_refine(input_string):
   # print("entered rewrite_and_refine")

    pattern = r"(REWRITTEN QUERY:.*?\?)"

    
  
    matches = re.findall(pattern, input_string, re.DOTALL)

    S = ''
    for match in matches:
        S += match + "\n"

    S = S.replace('?', "?\n")
    return  S

def plan_and_sched_txt(input_string):
    docs = []

    chunks = input_string.split("Document Summary:")
    thought = chunks[0]
    chunks = chunks[1:]
    docs.append({
        "thought": thought
    })

    return docs


def llm_answerer(question):
    check = check_guardrails(question)
    if check:
        return json.dumps({"state_messages": [], "final_ans": "Sorry, I can't answer this as the query violates the terms of service and ethical guidelines."})
    else:
        try:
            #print("Entered llm_answerer")
            with open("..\\legal-chatbot-backend\\text_files\\plan_and_schedule.txt", "w") as f:
                f.write("")
            with open("..\\legal-chatbot-backend\\text_files\\join.txt", "w") as f:
                f.write("")
            with open("..\\legal-chatbot-backend\\text_files\\human_input_node.txt", "w") as f:
                f.write("")
            with open("..\\legal-chatbot-backend\\text_files\\rewrite.txt", "w") as f:
                f.write("")
            with open("..\\legal-chatbot-backend\\text_files\\generate.txt", "w") as f:
                f.write("")
            answer = []
            for step in chain.stream({"messages": [HumanMessage(content=question)], }, config={"callbacks": [langfuse_handler]}):
                answer.append(step)

                # Optionally, write the step.pretty_print() output if you want that
                # step.pretty_print()

                for k, v in step.items():
                    # Write the output to the file
                    # pprint.pprint(f"Output from node '{k}':\n")
                    # pprint.pprint("---\n")
                    # pprint.pprint(v, indent=2, width=80, depth=None)
                    for message in step[k]["messages"]:
                        key = step[k]
                        #print("key: ", k)
                       # print("key: ", k)
                        if k == "plan_and_schedule":
                            M = plan_and_sched_txt(message.content)
                            if M:
                                with open("..\\legal-chatbot-backend\\text_files\\plan_and_schedule.txt", "a", encoding="utf-8") as f:
                                    f.write("Combining query with retrieved info....\n")

                        elif k == "join":
                            M = join_txt(message.content)
                            if M:
                                with open("..\\legal-chatbot-backend\\text_files\\join.txt", "a", encoding="utf-8") as f:
                                    s = str(M)
                                    s = s.replace('{\n', '')
                                    s = s.replace('}\n', '')
                                    s = s.replace('[\n', '')
                                    s = s.replace(']\n', '')
                                    s = s.replace('"thought": ', '')
                                    s = s.replace('},', '')
                                    s = s.replace(']', '')
                                    f.write(f"{str(s)}\n")

                        elif k == "human_input_node":
                            pass
                            # with open("..\\legal-chatbot-backend\\text_files\\human_input_node.txt", "a", encoding="utf-8") as f:
                            #     S = message.content
                            #     S = find_last_text_in_quotes(S)
                            #     f.write(f"{S}\n")

                        elif k == "rewrite":
                            if 'REWRITTEN QUERY' in message.content:
                                print("entered rewrite", message)
                                print("message.content: ", message.content)
                                M = rewrite_and_refine(message.content)
                                print("M: ", M)
                                # M = rewrite_and_refine2(M)
                                with open("..\\legal-chatbot-backend\\text_files\\rewrite.txt", "a", encoding="utf-8") as f:
                                    f.write(f"{str(M)}\n")

                        elif k == "generate":
                            with open("..\\legal-chatbot-backend\\text_files\\generate.txt", "a", encoding="utf-8") as f:
                                f.write(f"{message}\n")

                # pprint.pprint("\n---\n")


            langfuse_handler.langfuse.flush()

        except:
            print("Entered llm_answerer with exception")
            traceback.print_exc() 
            print('--------------------------------------------------------------------')

            with open("..\\legal-chatbot-backend\\text_files\\plan_and_schedule.txt", "w") as f:
                f.write("")
            with open("..\\legal-chatbot-backend\\text_files\\join.txt", "w") as f:
                f.write("")
            with open("..\\legal-chatbot-backend\\text_files\\human_input_node.txt", "w") as f:
                f.write("")
            with open("..\\legal-chatbot-backend\\text_files\\rewrite.txt", "w") as f:
                f.write("")
            with open("..\\legal-chatbot-backend\\text_files\\generate.txt", "w") as f:
                f.write("")
            answer = []
            for k, v in step.items():
                    # Write the output to the file
                    # pprint.pprint(f"Output from node '{k}':\n")
                    # pprint.pprint("---\n")
                    # pprint.pprint(v, indent=2, width=80, depth=None)
                    for message in step[k]["messages"]:
                        key = step[k]
                        #print("message in step k: ", message)
                        #print("key: ", k)
                        if k == "plan_and_schedule":
                            M = plan_and_sched_txt(message.content)
                            if M:
                                with open("..\\legal-chatbot-backend\\text_files\\plan_and_schedule.txt", "a", encoding="utf-8") as f:
                                    f.write(f"{str(M)}\n")

                        elif k == "join":
                            M = join_txt(message.content)
                            if M:
                                with open("..\\legal-chatbot-backend\\text_files\\join.txt", "a", encoding="utf-8") as f:
                                    f.write(f"{str(M)}\n")

                        elif k == "human_input_node":
                            print("content of human_input_node: ", step[k])
                            # print("last message: ", step[k]["messages"][-1])
                            # with open("..\\legal-chatbot-backend\\text_files\\human_input_node.txt", "a", encoding="utf-8") as f:
                            #     f.write(f"{step[k]}\n")

                        elif k == "rewrite":
                            M = rewrite_and_refine(message.content)
                            if M:
                                with open("..\\legal-chatbot-backend\\text_files\\rewrite.txt", "a", encoding="utf-8") as f:
                                    f.write(f"{str(M)}\n")

                        elif k == "generate":
                            with open("..\\legal-chatbot-backend\\text_files\\generate.txt", "a", encoding="utf-8") as f:
                                f.write(f"{message}\n")

                #pprint.pprint("\n---\n")
        
        final_answer = get_citations_with_ans(question, answer)
        ls = []
        for dict in answer:
            for k,v in dict.items():
                temp_ls = [message.content if hasattr(message, "content") else message for message in dict[k]['messages']]
                ls.append({k : temp_ls})
        #print({"state_messages":ls, "final_ans":final_answer})
        final_answer = final_answer.encode('utf-8').decode('utf-8')
        return json.dumps({"state_messages":ls, "final_ans":final_answer})

    

def store_in_mongodb(response, username, user_message, chat_num):
    # print("Entered store function: ", type(response))

    try:
        # Prepare the document to be inserted into MongoDB
        document = {
            "username": username,
            "scriptResponse": response,
            "usermessage": user_message,
            "chat_num": chat_num,  # Include chat_num here
            "createdAt": datetime.datetime.now()
        }

        # Insert the document into MongoDB
        collection.insert_one(document)
        print("Response stored in MongoDB")
    except Exception as e:
        print(f"Error storing in MongoDB: {e}")


if __name__ == "__main__":
    user_query = sys.argv[1]
    user_name = sys.argv[2]
    chat_id = int(sys.argv[3])  # Get the chat_id from the arguments and convert to int

    raw_response = llm_answerer(user_query)
    response = json.loads(raw_response.encode("utf-8", "replace").decode("utf-8"))


    #print("Response generated:", response)

    if response:
        store_in_mongodb(response, user_name, user_query, chat_id)  # Store the response and chat_id in MongoDB
    else:
        print("No response generated.")
