from src.data import load_data
from src.rag_model import RAGModel

def main():
    rag = RAGModel(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
               generation_model_name="google/flan-t5-small")
    headers = {'User-Agent': 'Mozilla/5.0'}
    catalogue_url = '/catalogue/period/18th%20century/region/London/source_type/Primary%20sources/subject/Economic'
    data = load_data(headers,catalogue_url)
    query = input("Enter your query: ")
    response = rag.process_query(query, data)
    print(f"Response: {response}")

if __name__ == '__main__':
	main()