from imports import *
from tools import *
from output_parser import LLMCompilerPlanParser, Task
from task_fetching_unit import *
import globals_

def create_planner(
    llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
):
    tool_descriptions = "\n".join(
        f"{i+1}. {tool.description}\n"
        for i, tool in enumerate(
            tools
        )  # +1 to offset the 0 starting index, we want it count normally from 1.
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools)
        + 1,  # Add one because we're adding the join() tool at the end.
        tool_descriptions=tool_descriptions,
    )

    replanner_prompt = base_prompt.partial(
        replan=' - You are given the results of the "Previous Plan" which is the plan that the previous agent created along with the execution results '
        "in the form of relevants documents retrieved in the last plan that answered a part of the user query."
        'You MUST use these information to create the next plan under "Current Plan". Remember that these documents only answer PART OF the user query. You have to ensure you make a plan to get the answer to the rest of the query that the previous retrieved documents do not answer.\n'
        ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
        'In the query given to a retriever tool, you should '
        " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list):
        # Context is passed as a system message
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        new = state[-1].content.split("- Begin counting at : ")
        state[-1].content = new[0]

        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break

        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": [state[-1]] + [state[-2]]}


    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )