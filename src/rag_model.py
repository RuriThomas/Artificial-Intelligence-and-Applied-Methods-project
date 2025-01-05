from traitlets import This
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import torch
from nltk.corpus import stopwords
import pickle
import os

class RAGModel:
    def __init__(self, embedding_model_name, generation_model_name):
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
        self.gen_model =  AutoModelForSeq2SeqLM.from_pretrained(generation_model_name)
        self.stopwords = set(stopwords.words('english'))


    def embed_texts(self,texts):
        texts = [self.normalize_text(text) for text in texts]
        inputs = self.embedding_tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            embeddings = self.embedding_model(**inputs).last_hidden_state.mean(dim=1)  
        return embeddings


    def retrieve_context(self, query, context_texts, context_embeddings, top_k=3):
        query_embedding = self.embed_texts([query])
        similarities = cosine_similarity(query_embedding, context_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [context_texts[i] for i in top_indices]

    def generate_response(self, query, context):
        input_text = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        inputs = self.gen_tokenizer.encode(input_text, return_tensors="pt", truncation=True)
        outputs = self.gen_model.generate(inputs, max_length=300,top_k=50)
        return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def process_query(self,query,save_path):
        context_texts, context_embeddings = self.load_embeddings(save_path)
        retrieved_context = self.retrieve_context(query, context_texts, context_embeddings)
        combined_context = " ".join(retrieved_context)
        response = self.generate_response(query, combined_context)
        return response
    
    def normalize_text(self,text):
        text = " ".join(text.split())
        words = text.split()
        text = " ".join([word for word in words if word not in self.stopwords])
        return text

    def extractDataIntoText(self,data): # Cannot chunk because it makes the embeding takes ages
        return [f"{entry['Title']}: {entry['Content']}" for entry in data]
        
    def precompute_embedding(self, data, save_path):
        if os.path.exists(save_path):
            return
        context_texts = self.extractDataIntoText(data)
        context_embeddings = self.embed_texts(context_texts)
        print("saving...")
        with open(save_path, "wb") as f:
            pickle.dump({"texts": context_texts, "embeddings": context_embeddings.numpy()}, f)
    
    def load_embeddings(self, save_path):
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        context_texts = data["texts"]
        context_embeddings = torch.tensor(data["embeddings"])
        return context_texts, context_embeddings
    
    def chunk_text(self, text, chunk_size=300):
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

