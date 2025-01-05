from src.data import load_data, saveLocal
from src.rag_model import RAGModel

_cb6 = '/search/series/House of Lords, Journals'
_743 = '/catalogue/period/19th century/subject/Parliamentary' 
def main():
    rag = RAGModel(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
               generation_model_name="google/flan-t5-small")
    headers = {'User-Agent': 'Mozilla/5.0'}
    catalogue_url = '/catalogue/period/19th century/subject/Parliamentary'
    data = load_data(headers,catalogue_url)
    saveLocal(catalogue_url,data)
    save_path = input("Enter save location of precomputed embeding: ")
    rag.precompute_embedding(data,save_path)
    while True:
        query = input("Enter your query: ")
        response = rag.process_query(query, save_path)
        print(f"Response: {response}")

if __name__ == '__main__':
	main()