from imports import *

from joiner import _parse_joiner_output
from utils import *
from tools import *
from prompts import *
from planner import *
from task_fetching_unit import *
from joiner import *
import globals_
from agents import *
from HIL import *
from HIL import _parse_feedback_output
from citations import *
import itertools
import copy
import sys
import warnings
warnings.filterwarnings('ignore')

load_dotenv()


def main(question):

    tools=[Master_tool]

    planner = create_planner(llm_with_fallback.call(), tools, prompt)



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

    check = check_guardrails(question)

    if check:
        print({"state_messages": [], "final_ans": "Sorry, I can't answer this as the query violates the terms of service and ethical guidelines."})
    else:
        try:
            answer = []
            for step in chain.stream({"messages": [HumanMessage(content=question)], }, config={"callbacks": [langfuse_handler]}):
                answer.append(step)

                for key, value in step.items():
                    # Write the output to the file
                    pprint.pprint(f"Output from node '{key}':\n")
                    pprint.pprint("---\n")
                    pprint.pprint(value, indent=2, width=80, depth=None)
                pprint.pprint("\n---\n")
            langfuse_handler.langfuse.flush()

        except:
            answer = []
            for step in chain.stream(
                {"messages": [HumanMessage(content=question)], }
            ):
                print(globals_.counter_)
                answer.append(step)
                print("---")
                for key, value in step.items():
                    pprint.pprint(f"Output from node '{key}':")
                    pprint.pprint("---")
                    pprint.pprint(value, indent=2, width=80, depth=None)
                pprint.pprint("\n---\n")
            
        # print(answer[-1]["generate"]["messages"][0])

        final_answer = get_citations_with_ans(question, answer)
        ls = []
        for dict in answer:
            for k,v in dict.items():
                temp_ls = [message.content if hasattr(message, "content") else message for message in dict[k]['messages']]
                ls.append({k : temp_ls})
        print({"state_messages":ls, "final_ans":final_answer}) 


if __name__=="__main__":
    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("No query provided!!") 