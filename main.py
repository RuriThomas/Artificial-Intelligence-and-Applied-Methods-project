from src.data import load_data, saveLocal
from src.rag_model import RAGModel

def main():
    rag = RAGModel(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
               generation_model_name="google/flan-t5-small")
    headers = {'User-Agent': 'Mozilla/5.0'}
    catalogue_url = '/vch/berks/vol4'
    data = load_data(headers,catalogue_url)
    saveLocal(catalogue_url,data)
    rag.precompute_embedding(data,"test2")
    query = input("Enter your query: ")
    response = rag.process_query(query, "test2")
    print(f"Response: {response}")

if __name__ == '__main__':
	main()