from imports import *
from anthropic_functions import ChatAnthropic
import globals_
from langfuse.callback import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host=os.getenv("LANGFUSE_HOST")
)

class LLMWithFallback:
    def __init__(self,primary_llm_class, primary_kwargs, fallback_llm_class, fallback_kwargs):
        """
        Initialize the primary and fallback LLMs, testing for API key validity.
        """
        self.primary_llm = None
        self.fallback_llm = None
        self.flag_primary=True
        self.flag_fallback=True
        # Try initializing the primary LLM
        try:
            print("Initializing primary LLM: ChatOpenAI...")
            self.primary_llm = primary_llm_class(**primary_kwargs)
            # Test API key validity with a small request
            # print("---Before CALL---")
            self.primary_llm.invoke("hi")
            # print("---AFTER CALL---")
        except Exception as e:
            print(f"Primary LLM initialization failed: {e}")
            self.flag_primary=False

        # Try initializing the fallback LLM
        try:
            print("Initializing fallback LLM: ChatAnthropic...")
            self.fallback_llm = fallback_llm_class(**fallback_kwargs)
            # Test API key validity with a small request
            self.fallback_llm.invoke("Hi")
        except Exception as e:
            print(f"Fallback LLM initialization failed: {e}")
            self.flag_fallback=False
    
    def call(self,*args,**kwargs):
        if self.flag_fallback and not self.flag_primary:
            self.primary_llm=self.fallback_llm
        if not self.flag_fallback and not self.flag_primary:
            raise ValueError("Both primary and fallback LLM initialization failed. Check your API keys.")

        if self.flag_primary:
            try:
                return self.primary_llm
            except Exception as e:
                print(f"Primary LLM failed with error: {e}")
        
        if self.flag_fallback:
            try:
                return self.fallback_llm
            except Exception as e:
                print(f"Fallback LLM failed with error: {e}")
        
        raise RuntimeError("Both primary and fallback LLMs failed to respond.")

def convert_messages( messages: Sequence[BaseMessage],
) -> List[Union[SystemMessage, AIMessage, HumanMessage]]:
    """Merge runs of human/tool messages into single human messages with content blocks.""" 
    result = []
    # print("---CONVERT WALE MESSAGES---")
    # print(messages)
    for i in messages:
        if isinstance(i, HumanMessage):
            result.append(i)
        elif isinstance(i, SystemMessage):
            result.append(i)
        elif isinstance(i, AIMessage):
            result.append(i)
        else:
            # print("---ERROR IDHAR AA RHA HAI ")
            # print(i)
            content = i.content
            if isinstance(i, ToolMessage):
                name = i.name
                tool_call_id = i.tool_call_id
                new = AIMessage(content=content, additional_kwargs={'type': "function", "func_name": name, "tool_call_id": tool_call_id})

            elif isinstance(i, FunctionMessage):
                name = i.name
                tool_call_id = i.tool_call_id
                idx = i.additional_kwargs.get("idx", None)
                args = i.additional_kwargs.get("args", None)
                new = AIMessage(content=content, additional_kwargs={'type': "function", "func_name": name, "tool_call_id": tool_call_id, "idx": idx, "args": args})
                
            result.append(new)

    return result


def get_key(x):
    return x.page_content.split(' ')[0]


# Define LLM configurations

primary_llm_class = ChatOpenAI
primary_kwargs = {"model": "gpt-4o", "api_key": os.getenv("OPENAI_API_KEY")}  
fallback_llm_class = ChatAnthropic
fallback_kwargs = {"model": "claude-3-5-sonnet-20241022", "api_key": os.getenv("ANTHROPIC_API_KEY")}  


# Initialize the LLM with fallback mechanism
llm_with_fallback = LLMWithFallback(primary_llm_class, primary_kwargs, fallback_llm_class, fallback_kwargs)