from imports import *
from utils import *
import globals_

class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]

    context: str = Field(
        description="The retrieved context in the form of the COMPLETE documents that were retrieved that led to the decision. Basically, the full output of the previous tools should come here"
    )

def debug_output(llm_output):
    print("Output from LLM before parsing:", llm_output)
    print(vars(llm_output))
    return llm_output


def select_recent_messages(state) -> dict:
    global doc_keys

    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break


    regex = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    for i in selected:
        if isinstance(i, FunctionMessage):
            if i.content == "join":
                continue
            item = re.findall(regex, i.content)


    if (isinstance(llm_with_fallback.call(), ChatOpenAI)):
        new_messages=selected[::-1]
    elif(isinstance(llm_with_fallback.call(),ChatAnthropic)):
        new_messages = convert_messages(selected[::-1])
    return {"messages": new_messages}

def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    globals_.counter_+=1
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return {
            "messages": response
            + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}\n The Retrieved Documents are: {decision.context}"

                )
            ]
        }
    else:
        return {"messages": response + [AIMessage(content=decision.action.response)] + [AIMessage(content=f"Context from last attempt: {decision.context}")]}
