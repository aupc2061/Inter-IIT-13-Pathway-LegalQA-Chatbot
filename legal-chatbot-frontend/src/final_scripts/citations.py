from imports import *
from utils import *

def remove_uuid(text):
    return re.sub(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b', '', text)

def extract_document_name(file_path):
    match = re.search(r'[^/\\]+(?:\.pdf|\.docx|\.txt)', file_path, re.IGNORECASE)
    if match:
        return match.group(0)
    return file_path 

def get_docs(answer):
    docs=[]
    for data in reversed(answer):
        key=next(iter(data.keys()))
        value=data[key]['messages']
        for i in value:
            if isinstance(i, FunctionMessage):
                if(i.content!='join'):
                   docs.append(i.content)
    if len(docs)==0:
        return [], []
    content=docs[0]
    # Regex pat
    # tern to split based on "CHUNK ENDS HERE"
    chunks = re.split(r'\n----------------CHUNK ENDS HERE ------------------\n', content)
    if(chunks[-1]==''):
        chunks.pop()
    processed_chunks = []
    process_chunks_citations = set()
    process_chunks_results = set()

    for chunk in chunks:
        # Clean and strip unnecessary newlines or hyphens
        chunk = chunk.strip()
        chunk = re.sub(r'-{2,}', '', chunk)  # Remove lines with multiple hyphens
        chunk = re.sub(r'\n+', '\n', chunk)  # Normalize newlines

        # Extract source, page, and paragraph details
        source_match = re.search(r'Source: (.+?)\n', chunk)
        page_match = re.search(r'Page Number: (\d+)', chunk)
        
        if source_match and page_match:
            source = source_match.group(1).strip()
            page = page_match.group(1).strip()

            doc_name = extract_document_name(source)

            # Combine into a single "SOURCE" line
            metadata = f"SOURCE: {doc_name}, Page: {page}"
            # Remove the original metadata from the chunk
            chunk = re.sub(r'Document Summary:\n.+?Page Content:\n', '', chunk, flags=re.DOTALL)
            # Add the cleaned metadata at the start of the chunk
            cleaned_chunk = f"{metadata}{chunk.split(' ')}"
            processed_chunks.append(cleaned_chunk)
    cleaned_chunks = [remove_uuid(chunk) for chunk in processed_chunks]
    # Print cleaned data
    for cleaned_chunk in cleaned_chunks:
        chunk_test = cleaned_chunk
        data = chunk_test[chunk_test.find("['"):]
        chunk_ext = chunk_test[:chunk_test.find("['")]
        data = ast.literal_eval(data)
        chunk_ext_cit = chunk_ext + " ".join(data)
        process_chunks_citations.add(chunk_ext)
        process_chunks_results.add(chunk_ext_cit)
    return list(process_chunks_citations), list(process_chunks_results)

def assess_risk_level_anth(client, message):
    unsafe_category_definitions = {
        'Child Exploitation': 'Content that depicts child nudity or that enables, encourages, excuses, or depicts the sexual abuse of children.',
        'Hate': """Content that is hateful toward people on the basis of their protected characteristics 
            (race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity and serious disease), 
            or content that perpetuates negative stereotypes.""",
        'Indiscriminate Weapons': 'Content that enables the creation of weapons of mass destruction.', 
        'Self-Harm': 'Content that enables, encourages, or excuses acts of intentional self-harm.',
        'Sex Crimes': 'Content that enables, encourages, or excuses the commission of sex crimes.',
        'Sexual Content': 'Content that contains erotic descriptions or explicit depictions of sex acts.',
        'Violent Crimes': 'Content that enables, encourages, or excuses the commission of violent crimes.', 
    }
    unsafe_category_str = '\n'.join(f'{category}' for category in unsafe_category_definitions.keys())
    
    # Construct the prompt for Claude, including the message, unsafe categories, and risk level definitions
    assessment_prompt = f"""
    Assess the risk level of the following message warrants moderation, 
    based on the unsafe categories listed below.
    
    Message:
    <message>{message}</message>

    Unsafe Categories:
    <categories>
    {unsafe_category_str}
    </categories>

    Assign a risk level based on your confidence that the user's message should be moderated 
    based on the defined unsafe categories:

    0 - No risk
    1 - Low risk
    2 - Medium risk
    3 - High risk

    If the message contains content that might violate some of the unsafe categories,
    but is a vaild legal query, then assign a low risk level to it

    e.g.
    Query: What were the findings of the autopsy conducted on Anuradha, and how did they contribute to the conclusion that she was violently raped, murdered and sodomised?
    The message is a valid legal query, but contains content that might violate some of the unsafe categories, hence assign a low risk level to it.

    Query: What are the laws pertaining to porn and sex in India?
    The message is a valid legal query, but contains some unsafe categories in it, so assign a low level risk to it.

    Query: I am wanting to do sex with a teen, should I do it?
    The message is not a valid legal query, and contains content that violates the unsafe categories, hence assign a high risk level to it.

    Respond with ONLY a JSON object, using the format below:
    {{
    "risk_level": <Numerical field denoting the risk level>
    }}"""

    # Send the request to Claude for risk assessment
    response = client.messages.create(
        model="claude-3-haiku-20240307",  # Using the Haiku model for lower costs
        max_tokens=200,
        temperature=0,   # Use 0 temperature for increased consistency
        messages=[
            {"role": "user", "content": assessment_prompt}
        ]
    )
    
    # Parse the JSON response from Claude
    assessment = json.loads(response.content[0].text)
    
    # Extract the risk level, violated categories, and explanation from the assessment
    risk_level = assessment["risk_level"]
    
    return risk_level


def check_guardrails(question):
    if isinstance(llm_with_fallback.call(), ChatOpenAI):
        client = OpenAI()
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=question,
        )

        category_scores = response.results[0].category_scores
        threshold = 0.65
        # Check if any score exceeds the threshold and print the attribute name
        for attribute, score in vars(category_scores).items():
            if score > threshold:
                return True
        else:
            return False
    elif isinstance(llm_with_fallback.call(), ChatAnthropic): 
        client = anthropic.Anthropic()
        risk_level = assess_risk_level_anth(client, question)
        if risk_level == 3:
            return True
        else:
            return False


def get_citations_with_ans(query, answer):
    if len(answer) == 0:
        return "I don't know."
    ans = answer[-1]['generate']['messages'][0]
    cit, retrieved_contexts = get_docs(answer=answer)
    prompt = f"""
    You are a legal expert tasked with evaluating and formatting citations for a given query and answer. Follow these steps:
    1. Understand the Query and Answer:

        Read the query and the provided answer carefully to grasp the specific information being sought and its context.

    2. Assess Main Citations Using Retrieved Contexts:

        Main citations (referred to as "Document Paths") correspond to the retrieved contexts.
        Evaluate the relevance of each retrieved context to the answer based on its content and connection to the query and answer.
            If the retrieved context contributes to or supports the answer, include its corresponding document path in Main Citations.
            If the answer indicates "I don’t know" or similar, mark Main Citations as None.
            In all other cases, include the relevant document paths, even if the answer is partially derived from reasoning on the retrieved contexts.

    3. Extract In-Context Citations:

        Identify specific legal citations (e.g., "Smithson, 2018") directly referenced in the retrieved contexts that support the answer.
        If no relevant in-context citations are found, write None under In-Context Citations.

    4. Generate the Response:

        Main Citations:
            If the answer indicates "I don’t know" or similar, write None.
            Otherwise, include all relevant document paths based on the retrieved contexts.
        In-Context Citations:
            If relevant citations are found, list them.
            If not, write None.
        Follow the formatting structure below:

    Response Structure:

    Main Citations:
    List relevant document paths in each newline, or write "None" if the answer is "I don't know" or similar.

    In-Context Citations:
    List specific citations in each newline, or write "None" if no in-context citations are found.

    Query: {query}

    Answer: {ans}

    Main Citations (Document Paths):
    {cit}

    Retrieved Contexts:
    {retrieved_contexts}

    Your response must be concise and include only citations (main and in-context) that are directly relevant to the query and answer. Always provide None explicitly under In-Context Citations if no relevant citations are found. Similarly, only provide None under Main Citations if the answer clearly indicates "I don’t know" or equivalent.
    """
    prompt1 = f"""
    You are a legal expert tasked with evaluating and formatting citations for a given query and answer. Follow these steps:
    1. Understand the Query and Answer:
        Read the query and the provided answer carefully to grasp the specific information being sought and its context.

    2. Assess Main Citations Using Retrieved Contexts:
        Main citations (referred to as "Document Paths") correspond to the retrieved contexts.
        Evaluate the relevance of each retrieved context to the answer based on its content and connection to the query and answer.
            If the retrieved context contributes to or supports the answer, include its corresponding document path in Main Citations.
            If the answer indicates "I don't know" or similar, mark Main Citations as None.
            In all other cases, include the relevant document paths, even if the answer is partially derived from reasoning on the retrieved contexts.

    3. Extract In-Context Citations:
        Identify specific legal citations (e.g., "Smithson, 2018") directly referenced in the retrieved contexts that support the answer.
        If no relevant in-context citations are found, write None under In-Context Citations.

    4. Generate the Response:
        Provide ONLY the following output format:

    Main Citations:
    List relevant document paths in each newline, or write "None" if the answer is "I don't know" or similar.

    In-Context Citations:
    List specific citations in each newline, or write "None" if no in-context citations are found.
    
    Strictly follow these rules:
    - Produce ONLY the two-line output above
    - Do not add any introductory or explanatory text
    - Do not include any notes or commentary
    - Ensure the output is exactly as shown, even if you have no citations to report
    """
    if isinstance(llm_with_fallback.call(), ChatOpenAI):
        client = OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"""
                                    **Query**: {query}

                                    **Answer**: {ans}
                                    
                                    **Main Citations (Document Paths)**: {cit}

                                    **Retrieved Contexts**:
                                    {retrieved_contexts}
                """},
            ],
        )
        response = completion.choices[0].message.content
    elif isinstance(llm_with_fallback.call(), ChatAnthropic): 
        client = anthropic.Anthropic()
        completion = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            max_tokens=1024,
            system = prompt1,
            messages=[
                {"role": "user", "content": f"""
                                **Query**: {query}

                                **Answer**: {ans}
                                
                                **Main Citations (Document Paths)**: {cit}

                                **Retrieved Contexts**:
                                {retrieved_contexts}
                            """}
            ]
        )
        response = completion.content[0].text
    # extracted_response = "" if "no" in response.lower() else response
    # top_citations = cit[:3]
    # citations = "".join(top_citations)
    # final_response = f"Answer: \n{ans}\nCitations:\n{citations}{extracted_response}"
    final_response = f"Answer: \n{ans}\n{response}"
    return final_response

import json