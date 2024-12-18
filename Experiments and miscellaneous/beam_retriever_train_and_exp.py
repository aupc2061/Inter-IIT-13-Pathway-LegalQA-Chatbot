

# tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# def preprocess_data(fake_dataset, tokenizer, max_length=512):
#     input_ids_list, attention_mask_list, labels_list, classifier_types = [], [], [], []

#     for example in fake_dataset:
#         question = example["question"]
        
        
#         for candidate in example["candidate_passages"]:
#             candidate_text = candidate["text"]
#             label = candidate["label"]

            
#             input_sequence = f"{question} {candidate_text}"
#             tokenized = tokenizer(
#                 input_sequence,
#                 truncation=True,
#                 padding="max_length",
#                 max_length=max_length,
#                 return_tensors="pt",
#             )

#             input_ids_list.append(tokenized["input_ids"].squeeze())
#             attention_mask_list.append(tokenized["attention_mask"].squeeze())
#             labels_list.append(label)
#             classifier_types.append("classifier1")

        
#         if example["retrieved_passages"]:
#             retrieved_context = " ".join(example["retrieved_passages"])
#             for candidate in example["candidate_passages"]:
#                 candidate_text = candidate["text"]
#                 label = candidate["label"]

                
#                 input_sequence = f"{question} {retrieved_context} {candidate_text}"
#                 tokenized = tokenizer(
#                     input_sequence,
#                     truncation=True,
#                     padding="max_length",
#                     max_length=max_length,
#                     return_tensors="pt",
#                 )

#                 input_ids_list.append(tokenized["input_ids"].squeeze())
#                 attention_mask_list.append(tokenized["attention_mask"].squeeze())
#                 labels_list.append(label)
#                 classifier_types.append("classifier2")

 
#     input_ids = torch.stack(input_ids_list)
#     attention_mask = torch.stack(attention_mask_list)
#     labels = torch.tensor(labels_list)
#     return input_ids, attention_mask, labels, classifier_types


# input_ids, attention_mask, labels, classifier_types = preprocess_data(fake_dataset, tokenizer)


# %%
# class BeamRetriever(nn.Module):
#     def __init__(self, model_name):
#         super(BeamRetriever, self).__init__()
#         self.encoder = AutoModel.from_pretrained(model_name)
#         hidden_size = self.encoder.config.hidden_size
        
#         # Two classification heads
#         self.classifier1 = nn.Linear(hidden_size, 2)  # Fixed candidates
#         self.classifier2 = nn.Linear(hidden_size, 2)  # Variable candidates

#     def forward(self, input_ids, attention_mask, classifier_type="classifier1"):
#         outputs = self.encoder(input_ids, attention_mask=attention_mask)
#         cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token representation
        
#         if classifier_type == "classifier1":
#             logits = self.classifier1(cls_output)
#         elif classifier_type == "classifier2":
#             logits = self.classifier2(cls_output)
#         else:
#             raise ValueError("Invalid classifier_type. Choose 'classifier1' or 'classifier2'.")
        
#         return logits



# model = BeamRetriever("microsoft/deberta-v3-base")
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)


# dataset = TensorDataset(input_ids, attention_mask, labels)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# epochs = 10
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0

#     for i, (batch_input_ids, batch_attention_mask, batch_labels) in enumerate(dataloader):
#         classifier_type = classifier_types[i]  

      
#         logits = model(
#             input_ids=batch_input_ids,
#             attention_mask=batch_attention_mask,
#             classifier_type=classifier_type
#         )
#         loss = criterion(logits, batch_labels)
#         total_loss += loss.item()

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#     print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score

fake_dataset = [
    {
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "retrieved_passages": [],
        "candidate_passages": [
            {"text": "Arthur's Magazine started in 1844.", "label": 1},
            {"text": "First for Women started in 1989.", "label": 0},
        ],
    },
    {
        "question": "What is the capital of France?",
        "retrieved_passages": ["France is a country in Europe with a rich history."],
        "candidate_passages": [
            {"text": "The capital of France is Paris.", "label": 1},
            {"text": "Berlin is the capital of Germany.", "label": 0},
        ],
    },
]




class HotpotQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        context = " ".join([" ".join(p[1]) for p in example["context"]])
        label = 1 if example["answer"] in [p[0] for p in example["supporting_facts"]] else 0
        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base", num_labels=2)


with open("hotpot_dev_fullwiki_v1_reduced.json", "r") as f:
    data = json.load(f)

dataset = HotpotQADataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


optimizer = AdamW(model.parameters(), lr=1e-4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 1

from tqdm import tqdm
from sklearn.metrics import accuracy_score

for epoch in range(epochs):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    print(f"\nEpoch {epoch + 1}/{epochs}")
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Accuracy = {accuracy:.4f}")


# model.eval()
# with torch.no_grad():
#     all_preds = []
#     all_labels = []

#     for batch in dataloader:
  
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["label"].to(device)

#         outputs = model(input_ids, attention_mask=attention_mask)
#         logits = outputs.logits
#         preds = torch.argmax(logits, dim=-1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

#     accuracy = accuracy_score(all_labels, all_preds)
#     print(f"Evaluation Accuracy = {accuracy:.4f}")






# %%
# import json
# with open("hotpot_dev_fullwiki_v1.json", "r") as f:
#     data = json.load(f)

# if isinstance(data, list):
#     # Take the first 1/5th of the list
#     reduced_data = data[:len(data) // 5]
# else:
#     raise ValueError("Expected a list in the JSON file, but got a different structure.")

# with open("hotpot_dev_fullwiki_v1_reduced.json", "w") as f:
#     json.dump(reduced_data, f, indent=4)

# print("Reduced data saved to 'hotpot_dev_fullwiki_v1_reduced.json'")


from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = BertTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
model = BertForSequenceClassification.from_pretrained('nlpaueb/legal-bert-small-uncased', num_labels=2).to(device)


class RelevanceDataset(Dataset):
    def __init__(self, questions, texts, labels, tokenizer, max_len=512):
        self.questions = questions
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            question,
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten().to(device),
            'attention_mask': encoding['attention_mask'].flatten().to(device),
            'labels': torch.tensor(label, dtype=torch.long).to(device)
        }


def train_model(model, dataloader, optimizer, device):
    model = model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    model = model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

        
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return total_loss / len(dataloader), accuracy


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def prepare_data(json_data):
    questions = []
    texts = []
    labels = []

    for entry in json_data:
        question = entry['question']
        answer = entry['answer']
        irrelevant_text = entry['irrelevant_text']

    
        questions.append(question)
        texts.append(answer)
        labels.append(1)

        for text in irrelevant_text:
            questions.append(question)
            texts.append(text)
            labels.append(0)

    return questions, texts, labels


def train(json_file_path, epochs=150, batch_size=1, learning_rate=2e-7):
    json_data = read_json_file(json_file_path)
    questions, texts, labels = prepare_data(json_data)

   
    train_questions, val_questions, train_texts, val_texts, train_labels, val_labels = train_test_split(
        questions, texts, labels, test_size=0.2, random_state=42
    )

 
    train_dataset = RelevanceDataset(train_questions, train_texts, train_labels, tokenizer)
    val_dataset = RelevanceDataset(val_questions, val_texts, val_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

  
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(model, train_dataloader, optimizer, device)
        val_loss, val_accuracy = evaluate_model(model, val_dataloader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print('----------------------------')


json_file_path = 'data.json'




passages_dict = {
    "correct_answer": "Central Council shall elect the president and Vice president.",
    
    "incorrect_1": "Passage about legal context A.",
    "incorrect_2": "Passage discussing legal precedent B.",
    "incorrect_3": "Passage mentioning case C.",
    "incorrect_4": "This is set of case C.",
    
    "irrelevant_1": "The weather today is quite pleasant, perfect for a walk outside.",
    "irrelevant_2": "Here is a recipe for making chocolate chip cookies, it's a fan favorite.",
    "irrelevant_3": "The history of ancient Rome is fascinating, particularly the rise of the Roman Empire.",
    "irrelevant_4": "Did you know that the speed of light is approximately 299,792 kilometers per second?",
    "irrelevant_5": "Here's a fun fact: honey never spoils, it can last thousands of years.",
    "irrelevant_6": "Cats sleep for about 12–16 hours a day, making them one of the sleepiest animals.",
    "irrelevant_7": "Mount Everest is the highest mountain in the world, standing at 8,848 meters.",
    "irrelevant_8": "The Mona Lisa by Leonardo da Vinci is one of the most famous paintings in the world.",
    "irrelevant_9": "Blue whales are the largest animals ever to have lived on Earth.",
    "irrelevant_10": "A standard deck of playing cards has 52 cards divided into four suits.",
    "irrelevant_11": "Water boils at 100°C or 212°F at sea level.",
    "irrelevant_12": "The Great Wall of China is over 13,000 miles long and took centuries to build.",
    "irrelevant_13": "Bananas are berries, but strawberries are not, according to botanical definitions.",
    "irrelevant_14": "Sharks have been around for over 400 million years, predating dinosaurs.",
    "irrelevant_15": "The Eiffel Tower was initially intended to be a temporary structure for the 1889 World Fair.",
    "irrelevant_16": "Coffee is the second most traded commodity in the world after crude oil.",
    "irrelevant_17": "The Amazon Rainforest produces 20% of the world's oxygen supply.",
    "irrelevant_18": "The moon is slowly drifting away from Earth at a rate of about 3.8 cm per year.",
    "irrelevant_19": "Octopuses have three hearts and blue blood.",
    "irrelevant_20": "Albert Einstein won the Nobel Prize in Physics in 1921 for his explanation of the photoelectric effect.",
    "irrelevant_21": "The world's oldest known tree is a bristlecone pine estimated to be over 5,000 years old.",
    "irrelevant_22": "The human brain contains approximately 86 billion neurons.",
    "irrelevant_23": "Venus is the hottest planet in the solar system due to its thick, toxic atmosphere.",
    "irrelevant_24": "The first email was sent in 1971 by computer engineer Ray Tomlinson.",
    "irrelevant_25": "A group of flamingos is called a 'flamboyance.'"
}






def get_relevance_score(question, paragraph, model):
        inputs =model.tokenizer(
            question + " [SEP] " + paragraph,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move inputs to the correct device
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

    
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)
        
    
        score = torch.sigmoid(outputs).item()
        return score

from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")  # Example model
model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")


def score_passage(query, passage):

    inputs = tokenizer.encode_plus(query, passage, add_special_tokens=True, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
  
    with torch.no_grad():
        outputs = model(**inputs)
        

    logits = outputs.logits
    score = logits.softmax(dim=-1)  
    return score


passages_dict = {
    "correct_answer": "The consultants endorsement means that the consultant shall be paid a fee for the services rendered.",
    "incorrect_1": "Passage about legal context A.",
    "incorrect_2": "Passage discussing legal precedent B.",
    "incorrect_3": "Passage mentioning case C.",
    "incorrect_4": "This is set of case C.",
    "irrelevant_1": "The weather today is quite pleasant, perfect for a walk outside.",
    "irrelevant_2": "Here is a recipe for making chocolate chip cookies, it's a fan favorite.",
    "irrelevant_3": "The history of ancient Rome is fascinating, particularly the rise of the Roman Empire.",
    "irrelevant_4": "Did you know that the speed of light is approximately 299,792 kilometers per second?",
    
}

query = "What does the term 'CONSULTANT'S ENDORSEMENT' include under the ADAMGOLF agreement?"


scores = {}
for key, passage in passages_dict.items():
    score = score_passage(query, passage)
    scores[key] = score[0][1].item()  # Take the score for the 'relevant' class (assuming binary classification)


sorted_passages = scores.items()


for passage, score in sorted_passages:
    print(f"Passage: {passage}\nScore: {score}\n")

model.save_pretrained("fine_tuned_CUAD_and_INDIAN_ACTS")
tokenizer.save_pretrained("fine_tuned_CUAD_and_INDIAN_ACTS")



from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")  # Example model
model = BertForSequenceClassification.from_pretrained("allenai/scibert_scivocab_uncased")



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class ConsultantEndorsementModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def load(self, model_path: str, tokenizer_path: str):

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def score_passage(self, query: str, passage: str) -> torch.Tensor:

        inputs = self.tokenizer(query, passage, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def get_answer(self, query: str, passages_dict: dict) -> list:

        scores = {}
        for key, passage in passages_dict.items():
            score = self.score_passage(query, passage)
            scores[key] = score[0][1].item()  # Assuming binary classification: take the 'relevant' score
        
        sorted_passages = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_passages

    def save(self, model_path: str, tokenizer_path: str):
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)



    


model = ConsultantEndorsementModel()
model.load("fine_tuned_CUAD_and_INDIAN_ACTS", "fine_tuned_CUAD_and_INDIAN_ACTS")
query = "What does the term 'CONSULTANT'S ENDORSEMENT' include under the ADAMGOLF agreement?"
passages_dict = {
    "correct_answer": "The term consultant endorsmeent includes the right to use the CONSULTANT'S name, fame, nickname, autograph, voice, facsimile signature, photograph, likeness, and image in connection with marketing, advertising, promotion, and sale of ADAMS GOLF'S PRODUCTS.",
    "incorrect_1": "Passage about legal context A.",
    "incorrect_2": "Passage discussing legal precedent B.",
    "incorrect_3": "Passage mentioning case C.",
    "incorrect_4": "This is set of case C.",
    "irrelevant_1": "The weather today is quite pleasant, perfect for a walk outside.",
    "irrelevant_2": "Here is a recipe for making chocolate chip cookies, it's a fan favorite.",
    "irrelevant_3": "The history of ancient Rome is fascinating, particularly the rise of the Roman Empire.",
    "irrelevant_4": "Did you know that the speed of light is approximately 299,792 kilometers per second?",
    
}
sorted_passages = model.get_answer(query, passages_dict)

for passage, score in sorted_passages:
    print(f"Passage: {passage}\nScore: {score}\n")


# %%
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle

class SimilarityModel:
    def __init__(self, model_name="law-ai/InLegalBERT"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def get_embeddings(self, text):
        encoded_input = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            output = self.model(**encoded_input)

        last_hidden_state = output.last_hidden_state
        embeddings = last_hidden_state.mean(dim=1)  # Mean pooling
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
        """Loads the tokenizer and model from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls()
        instance.tokenizer = data['tokenizer']
        instance.model = data['model']
        return instance



similarity_model = SimilarityModel()

text1 = "What does the term 'CONSULTANT'S ENDORSEMENT' include under the ADAMGOLF agreement?"
passages_dict = {
    "correct_answer": "The term consultant endorsement includes the right to use the CONSULTANT'S name, fame, nickname, autograph, voice, facsimile signature, photograph, likeness, and image in connection with marketing, advertising, promotion, and sale of ADAMS GOLF'S PRODUCTS.",
    "incorrect_1": "Passage about legal context A.",
    "incorrect_2": "Passage discussing legal precedent B.",
    "incorrect_3": "Passage mentioning case C.",
    "incorrect_4": "This is set of case C.",
    "irrelevant_1": "The weather today is quite pleasant, perfect for a walk outside.",
    "irrelevant_2": "Here is a recipe for making chocolate chip cookies, it's a fan favorite.",
    "irrelevant_3": "The history of ancient Rome is fascinating, particularly the rise of the Roman Empire.",
    "irrelevant_4": "Did you know that the speed of light is approximately 299,792 kilometers per second?",
}

for label, passage in passages_dict.items():
    similarity_score = similarity_model.score(text1, passage)
    print(f"Similarity Score for {label}: {similarity_score}")


similarity_model.save('similarity_model.pkl')
loaded_model = SimilarityModel.load('similarity_model.pkl')
similarity_score_loaded_model = loaded_model.score(text1, passages_dict["correct_answer"])
print(f"Loaded Model Similarity Score: {similarity_score_loaded_model}")


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
        embeddings = last_hidden_state.mean(dim=1)  # Mean pooling
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
        """Loads the tokenizer and model from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        instance = cls()
        instance.tokenizer = data['tokenizer']
        instance.model = data['model']
        return instance



similarity_model = BEAM_HEAD()


similarity_model.load('fine_tuned.pkl')


q2 = 'The club ins the best place to find a lover so the bar is where I go me asnd my friends at the table doing shots drinking fast and we talk slow'
q1 = 'wHAT ARE THE INDIAN LAWS RELATING TO THE CONSULTANT ENDORSEMENT'
print(similarity_model.compute_similarity(q1,q2))

head = BEAM_HEAD()

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
            
                
Beam = BeamRetriever(head, 3, 3)
D = {
  "node1": "The defendant has the right to a fair trial under the Sixth Amendment.",
    "node2": "No person shall be deprived of life, liberty, or property, without due process of law.",
    "node3": "Innocent until proven guilty is a fundamental principle in the criminal justice system.",
    "node4": "The right to remain silent is a safeguard against self-incrimination.",
    "node50": "The Transfer of Property Act regulates the transfer of immovable property, including sales, mortgages, leases, and gifts.",
    "answer": "The term consultants endorsement includes the right to use the CONSULTANT'S name, fame, nickname, autograph, voice, facsimile signature, photograph, likeness, and image in connection with marketing, advertising, promotion, and sale of ADAMS GOLF'S PRODUCTS.",
    "answer1" : 'Adam golf made the agreeement with the consultant to use his name and fame for the promotion of the product',
    "answer2" : "The consultant endorsement requires the consultant to provide his name, fame, and likeness for the promotion of the product in the Adamgolf agreement",
}

x = Beam.retrieve("What does the term 'CONSULTANT'S ENDORSEMENT' include under the ADAMGOLF agreement?", D)
            



from beam_retriever import BeamRetriever,BEAM_HEAD
from langchain_community.vectorstores import PathwayVectorClient
client = PathwayVectorClient(host="127.0.0.1", port=8745)
def Beam_answerer(question, client):
  head = BEAM_HEAD()
  Beam = BeamRetriever(head,B=3, K=3)
  List = client.similarity_search_with_relevance_scores(question, k = 30)
  D = {}
  i = 0
  for x in List:
    D['node' + str(i)] = x[0].page_content[36:].strip()
    i+=1
  
  beam_nodes = Beam.retrieve(question, D)
  s = ''
  for nodes in beam_nodes:
    s += D[nodes]
    s += '\n'
  return s

Beam_answerer('Quality control in stremick heritage foods and PREMIER NUTRITION CORPORATION agreement', client)


