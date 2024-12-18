from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle
class BEAM_HEAD:
    def __init__(self, model_name="law-ai/InLegalBERT"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, text):
        encoded_input = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**encoded_input)

        last_hidden_state = output.last_hidden_state
        embeddings = last_hidden_state.mean(dim=1)  
        return embeddings

    def compute_similarity(self, text1, text2):
        embeddings1 = self.get_embeddings(text1)
        embeddings2 = self.get_embeddings(text2)
        similarity = F.cosine_similarity(embeddings1, embeddings2)
        return similarity.item()

    def score(self, text1, text2):
   
        return self.compute_similarity(text1, text2)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'tokenizer': self.tokenizer, 'model': self.model}, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls()
        instance.tokenizer = data['tokenizer']
        instance.model = data['model']
        return instance



class BeamRetriever:
    def __init__(self, head, B = 1, K = 2):
        self.head = head
        self.B = B
        self.K = K

    def retrieve(self, question, passages):
        S1 = set()
        K = self.K
        chains = []
        ids = set()
        for i in range(K+1):
            if i == 0:
                scores = [(self.head.compute_similarity(question, passages[ids]), ids) for ids in passages]
                scores = sorted(scores, key=lambda x: x[0], reverse=True)
                scores = scores[:self.B]
                chain = []
                for x in scores:
                    chain.append(([x[1]], x[0]))
                    ids.add(x[1])
                    S1.add(x[1])
                chains.append(chain)
                chains = chains[0]
                
                continue 

            temp = []


            for id in passages:
                if id in ids:
                    continue
                for chain in chains:
                    
                    score = chain[1]
                    nodes_l = chain[0]

                    temp.append((nodes_l, score, id)) 

            temp1 = []
            for x in temp:
                string = ''
                score = x[1]
                if x[2] in x[0]:
                    continue
                for node in x[0]:
                    string += passages[node] + ' '

                string += passages[x[2]]
                new_score = self.head.compute_similarity(question, string)
                score = score + new_score
                temp1.append((x[0]+[x[2]], score))

            
            sorted_temp = sorted(temp1, key=lambda x: x[1], reverse=True)
            chains = sorted_temp[:self.B]
        
        for x in chains:
            for y in x[0]:
                S1.add(y)
        return S1
