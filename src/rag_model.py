from traitlets import This
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity
import torch

class RAGModel:
    def __init__(self, embedding_model_name, generation_model_name):
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
        self.gen_model =  AutoModelForSeq2SeqLM.from_pretrained(generation_model_name)


    def embed_texts(self,texts):
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
        outputs = self.gen_model.generate(inputs, max_length=200, num_beams=3)
        return self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def process_query(self,query,data):
        context_texts = [f"{entry['Title']}: {entry['Content']}" for entry in data]
        context_embeddings = self.embed_texts(context_texts)
        retrieved_context = self.retrieve_context(query, context_texts, context_embeddings)
        combined_context = " ".join(retrieved_context)
        response = self.generate_response(query, combined_context)
        return response


