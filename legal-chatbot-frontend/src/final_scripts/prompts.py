from imports import *
from utils import *

document_prompt = PromptTemplate.from_template("""
Document Summary:
-----------------
Source: {source}

Page Content:
{page_content}

-----------------
""")

if isinstance(llm_with_fallback.call(), ChatOpenAI):
    prompt = hub.pull("poohthebear/planner_prompt")
    joiner_prompt = hub.pull("poohthebear/joiner_prompt").partial(
        examples=""
    )
    feedback_prompt = hub.pull("poohthebear/human_prompt").partial(
        examples=""
    ) 
elif isinstance(llm_with_fallback.call(), ChatAnthropic):
    prompt = hub.pull("var-alpha/testing_anth") 
    joiner_prompt =hub.pull("poohthebear/joiner_prompt_anth").partial(
        examples=""
    )
    feedback_prompt = hub.pull("poohthebear/human_prompt_anth").partial(
        examples=""
    )
