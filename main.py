from src.data import load_data
def main():
    headers = {'User-Agent': 'Mozilla/5.0'}
    catalogue_url = '/catalogue/region/North/source_type/Primary%20sources/subject/Local'
    data = load_data(headers,catalogue_url)
    for entry in data:
        print(f"Title: {entry['Title']}\nContent: {entry['Content']}\n")

if __name__ == '__main__':
	main()