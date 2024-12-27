import requests
from bs4 import BeautifulSoup

def load_data(url):
    headers = {'User-Agent': 'Mozilla/5.0'} 
    catalogueUrl = '/catalogue/source_type/Primary%20sources'
    response = requests.get(url+catalogueUrl, headers=headers)
    response.raise_for_status()  

    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table')
    catalogue_table = tables[0]
    rows = catalogue_table.find_all('tr')
    catalogue_entries = []

    for row in rows:
        columns = row.find_all('td')
        if columns:
            title = columns[0].get_text(strip=True)
            link = columns[0].find('a')['href'].strip()
            catalogue_entries.append({'Title': title, 'Link': link})

    for entry in catalogue_entries:
        print(f"Title: {entry['Title']}, Link: {entry['Link']}")
    return catalogue_entries

