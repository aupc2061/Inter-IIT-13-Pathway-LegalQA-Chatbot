from imports import *
from joiner import *

### This condition determines looping logic
class HumanFeedback(BaseModel):
    """Extract whether to replan or generate based on the human input."""
    """Examples: 1) Replan - go ahead with replanning"""
    """          2) Generate - Do not replan and give FinalResponse"""
    """          3) Do not Replan - Do not replan and give FinalResponse"""

    # thought: Optional[str] = Field(
    #     default="I need to replan since the documents retrieved do not answer the query",
    #     description="The chain of thought reasoning for the selected action."
    # )
    # action: Union[FinalResponse, Replan]
    action: Union['FinalResponse', 'Replan'] = Field(
        description="Action to take: 'Replan' to replan, 'Generate' to return the final response without replanning."
    )

def _parse_feedback_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought:")]
    if isinstance(decision.action, Replan):
        return {
            "messages": response
            + [
                SystemMessage(
                    content=f"Context from last attempt: {decision.action.feedback}\n"

                )
            ]
        }
    else:
        return {"messages": response + [AIMessage(content=decision.action.response)]}
    
def select_last_message(state) -> dict:
    messages = state["messages"]
    human_query = SystemMessage(content= "ORIGINAL HUMAN QUERY: " + messages[0].content)
    human_instructions = HumanMessage(content= "ACTION TO BE TAKEN BASED ON HUMAN INPUT: " + messages[-1].content)
    context = HumanMessage(content= "CONTEXT FROM LAST ATTEMPT: " + messages[-2].content)
    ORDERS = {"messages": [human_query]+ [human_instructions]+ [context]}
    return {"messages":[human_instructions]}

