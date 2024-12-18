import os
import google.generativeai as genai
import re
import os
import time
from unstructured.partition.pdf import partition_pdf
os.environ['GEMINI'] = 'ENTER GEMINI API KEY HERE'
genai.configure(api_key=os.environ['GEMINI'])


def extract_integers(s: str) -> list[int]:
    return list(map(int, re.findall(r'\d+', s)))


model = genai.GenerativeModel("gemini-1.5-flash-8b")
elements = partition_pdf("b.pdf")

def lumber_chunker(parsed_data_list, theta=500):
    flag = True
    new_list = []
    genai.configure(api_key=os.environ['GEMINI'])
    model = genai.GenerativeModel("gemini-1.5-flash-8b")

    PROMT = ''' You have been given a list of chunk id, its type and its content.
     You have to return the chunk id of that chunk where the meaning of the paragraph changes drastically. 
     If there is no such chunk, return the chunk id of the last chunk.
    Each chunk will be of the form : 
    Chunk id : The id of the chunk
    Type of chunk : The type of chunk(HEADER, FOOTER, etc)
    TEXT : The actual content of the chunk
    Be concise. Just give the chunk id.
     
     '''
    counter = -1
    n = len(parsed_data_list)

    while counter < n:
        tokens = 0
        query = ''
        temp = counter
        while tokens < theta:
            temp += 1
            if temp >= n: 
                break
            chunk = parsed_data_list[temp]
            query += '----------------\n'
            query += f'Chunk id : {temp}\n'
            query += f'Type of chunk : {chunk.category}\n'
            query += f'TEXT :\n{chunk.text}\n'
            query += '----------------\n'
            tokens += len(chunk.text)

        prompt = PROMT + query
        print('Query:', query)
        print('-----------------------------------------------')
        while True:
            try:
                response = model.generate_content(prompt)
                response_text = response.text
                break  
            except Exception as e:
        
                print("API call limit exceeded. Retrying in 30 seconds...")
                time.sleep(30)

        print("Response:", response_text)
        print('------------------')
        chunk_id = extract_integers(response_text)[0]
        new_chunk = ''

        for i in range(counter + 1, chunk_id + 1): 
            chunk = parsed_data_list[i]
            new_chunk += f'Type: {chunk.category}\n'
            new_chunk += f'{chunk.text}\n'

        new_list.append(new_chunk)
        print("Processed Chunk ID:", chunk_id)
        print('#######################################################')

        counter = chunk_id
        if counter > 80:
            return new_list

    return new_list


