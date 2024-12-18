
# !git clone https://github.com/IAAR-Shanghai/Meta-Chunking.git
# !pip install --upgrade huggingface_hub
# !pip show nvidia-cublas-cu12
# !pip show nvidia-cuda-cupti-cu12
# !pip show nvidia-cuda-nvrtc-cu12==12.1.105
# !pip show nvidia-cuda-runtime-cu12==12.1.105
# !pip show nvidia-cudnn-cu12==9.1.0.70
# !pip show nvidia-cufft-cu12==11.0.2.54
# !pip show nvidia-curand-cu12==10.3.2.106
# !pip show nvidia-cusolver-cu12==11.4.5.107
# !pip show nvidia-cusparse-cu12==12.1.0.106
# !pip show nvidia-ml-py==12.555.43
# !pip show nvidia-nccl-cu12==2.20.5
# !pip show nvidia-nvjitlink-cu12==12.5.82
# !pip show nvidia-nvtx-cu12
# !pip install auto-gptq
# !pip install --upgrade optimum
# !pip install --upgrade git+https://github.com/huggingface/transformers.git
# !pip install --upgrade accelerate
# !pip install -r requirements.txt
# !pip install context_cite
import os
current_dir = os.getcwd()
target_dir = os.path.join(current_dir, 'Meta-Chunking')
os.chdir(target_dir)
import nltk
nltk.download('punkt')
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from optimum.gptq import GPTQQuantizer, load_quantized_model
import json
import torch.nn.functional as F
from chunk_rag import extract_by_html2text_db_nolist, split_text_by_punctuation
from docling.document_converter import DocumentConverter


target_dir1 = os.path.join(target_dir, 'example')
os.chdir(target_dir1)
os.getcwd()
model_name_or_path = 'Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4'
device_map = "auto"

print("Loading model and tokenizer...")
small_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
small_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, device_map=device_map)

small_model.eval()
print("Model loaded successfully.")


def get_prob_subtract(model, tokenizer, sentence1, sentence2, language):
    if language == 'zh':
        query = '''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
        1. 将“{}”分割成“{}”与“{}”两部分；
        2. 将“{}”不进行分割，保持原形式；
        请回答1或2。'''.format(sentence1 + sentence2, sentence1, sentence2, sentence1 + sentence2)
    else:
        query = '''This is a text chunking task. You are a text analysis expert. Please choose one of the following two options based on the logical structure and semantic content of the provided sentence:
        1. Split "{}" into "{}" and "{}" two parts;
        2. Keep "{}" unsplit in its original form;
        Please answer 1 or 2.'''.format(sentence1 + ' ' + sentence2, sentence1, sentence2, sentence1 + ' ' + sentence2)

    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(query)
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    input_ids = prompt_ids
    output_ids = tokenizer.encode(['1', '2'], return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[:, -1, :]
        token_probs = F.softmax(next_token_logits, dim=-1)
    next_token_prob_1 = token_probs[:, output_ids[0][0]].item()
    next_token_prob_2 = token_probs[:, output_ids[0][1]].item()

    return next_token_prob_2 - next_token_prob_1


def meta_chunking(original_text, base_model, language, ppl_threshold, chunk_length):
    chunk_length = int(chunk_length)
    if base_model == 'PPL Chunking':
        final_chunks = extract_by_html2text_db_nolist(original_text, small_model, small_tokenizer, ppl_threshold, language=language)
    else:
        full_segments = split_text_by_punctuation(original_text, language)
        tmp = ''
        threshold = 0
        threshold_list = []
        final_chunks = []

        for sentence in full_segments:
            if tmp == '':
                tmp += sentence
            else:
                prob_subtract = get_prob_subtract(small_model, small_tokenizer, tmp, sentence, language)
                threshold_list.append(prob_subtract)
                if prob_subtract > threshold:
                    tmp += ' ' + sentence
                else:
                    final_chunks.append(tmp)
                    tmp = sentence
            if len(threshold_list) >= 5:
                last_ten = threshold_list[-5:]
                avg = sum(last_ten) / len(last_ten)
                threshold = avg
        if tmp != '':
            final_chunks.append(tmp)

    merged_paragraphs = []
    current_paragraph = ""
    if language == 'zh':
        for paragraph in final_chunks:
            if len(current_paragraph) + len(paragraph) <= chunk_length:
                current_paragraph += paragraph
            else:
                merged_paragraphs.append(current_paragraph)
                current_paragraph = paragraph
    else:
        for paragraph in final_chunks:
            if len(current_paragraph.split()) + len(paragraph.split()) <= chunk_length:
                current_paragraph += ' ' + paragraph
            else:
                merged_paragraphs.append(current_paragraph)
                current_paragraph = paragraph
    if current_paragraph:
        merged_paragraphs.append(current_paragraph)
    return '\n\n'.join(merged_paragraphs)


source = r"/content/drive/MyDrive/Affiliate_Agreements/BIOAMBERINC_04_10_2013-EX-10.34-DEVELOPMENT AGREEMENT.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
parsed_doc = result.document.export_to_markdown()  # output: "## Docling Technical Report[...]"


def test_meta_chunking(original_text, base_model, language, ppl_threshold, chunk_length):
    print("Performing Meta-Chunking...")
    meta_chunks = meta_chunking(original_text, base_model, language, ppl_threshold, chunk_length)

    print("\n--- Meta-Chunking Results ---")
    for chunk in meta_chunks.split('\n\n'):
      print(chunk)
      print("------")

    v = len(meta_chunks.split('\n\n'))
    print(f"Meta-Chunk Count: {v}")
    s = sum(len(chunk.split()) for chunk in meta_chunks.split('\n\n')) / len(meta_chunks.split('\n\n'))
    print(f"Average Meta-Chunk Length: {s}")


original_text = parsed_doc


base_model = "PPl Chunking"  
language = "en"
ppl_threshold = 0.0
chunk_length = 150

test_meta_chunking(original_text, base_model, language, ppl_threshold, chunk_length)

retrieved_contexts = ['     C.“MANDATORY\n1. TERM\n” means the right to use the CONSULTANT’S name, fame, nickname, autograph,voice, facsimile, signature, photograph, likeness, and image in connection with the marketing, advertising, promotion and sale ofADAMS GOLF’S PRODUCT.B.“PRODUCT\n” shall mean the following ADAMS GOLF PRODUCTS that CONSULTANT mustexclusively play/use in all Champions/Senior Professional Golf Association (SPGA) and Professional Golf Association(PGA) events at all times: [***** ] Confidential Material redacted and filed separately with the Commission.',
 '\n    \n         ENDORSEMENT OF NON-COMPETITIVE PRODUCT\nUSE OF PRODUCT\nIt is particularly and expressly understood and agreed that if CONSULTANT shall find in his sincere best reasonablejudgment that the MANDATORY PRODUCT so supplied is not suitable for his use in tournament competition, then he shallpromptly notify ADAMS GOLF in writing of such fact and the reasons therefor. Thereafter, ADAMS GOLF shall have aperiod of thirty (30) days to either, at ADAMS GOLF’S sole discretion, supply CONSULTANT with MANDATORYPRODUCT that is acceptable to him or terminate the agreement. It is agreed that if the contract is terminated pursuant tothis paragraph, the compensation due CONSULTANT shall be prorated from the date this Agreement is terminated.Proration of compensation shall be determined on the same repayment schedule as provide in paragraph 8A below. [***** ] Confidential Material redacted and filed separately with the Commission. 4\nIf CONSULTANT endorses or promotes a non-competitive product and in that endorsement or promotion CONSULTANTwears, plays, uses, holds or is in any way associated with a product that would constitute PRODUCT as defined under thisAgreement, CONSULTANT shall use objectively reasonable best efforts to ensure that PRODUCT is an ADAMS GOLFPRODUCT and it shall not be altered or changed in appearance in the endorsement in any manner whatsoever without theexpress written consent of ADAMS GOLF. When endorsing a non-competitive product, under no circumstances shallCONSULTANT wear, play, use, hold or in any way be associated with an ADAMS GOLF competitor’s Product.7. CONSULTANT’S\nSATISFACTION OF MANDATORY PRODUCT\nE.Notwithstanding paragraphs 4A, 4B and 4C above, CONSULTANT shall not be required to wear ADAMS GOLF[*****] in [*****] ads.5. EXCLUSIVE\nDuring the term of this Agreement, CONSULTANT shall exclusively play/use the MANDATORY PRODUCT. (It isexpressly understood by the parties that CONSULTANT may play [* ****] clubs in the bag other than ADAMS GOLF clubsincluding, but not limited to, a putter by a manufacturer other than ADAMS GOLF but may not endorse those clubs and/or putter.)6. CONSULTANT’S',
 '\n    \n         A. During the term of this Agreement, ADAMS GOLF shall provide CONSULTANT with sufficient quantities of suchMANDATORY PRODUCTS for CONSULTANT’S use as CONSULTANT may reasonably need to fulfill hisobligations under this agreement. ADAMS GOLF shall pay all charges in connection with the delivery ofMANDATORY PRODUCTS to CONSULTANT.B.In addition to paragraph 17A above, ADAMS GOLF shall provide CONSULTANT with [*****] sets of clubs forCONSULTANT’S family and friends each calendar year of this Agreement. [***** ] Confidential Material redacted and filed separately with the Commission.']
retrieved_contexts_str = '\n\n'.join(retrieved_contexts)


prompt = f"""
{retrieved_contexts_str}
"""

from context_cite import ContextCiter

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
context = prompt
query = "What does the term 'CONSULTANT'S ENDORSEMENT' include under the ADAMGOLF agreement?"
cc = ContextCiter.from_pretrained(model_name, context, query, device="cpu")


cc.response


cc.get_attributions(as_dataframe=True, top_k=5)





