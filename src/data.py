import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests_cache


def fetch_entry_details(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    contents = []
    rows = soup.select('table:first-of-type tr')
    for row in rows:
        columns = row.find_all('td')
        if columns:
            title = columns[0].get_text(strip=True)
            contents.append(title)
    return contents

def load_data(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    catalogue_url = '/catalogue/source_type/Primary%20sources'
    response = requests.get(url + catalogue_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    rows = soup.select('table:first-of-type tr')
    catalogue_entries = []
    links = []

    for row in rows:
        columns = row.find_all('td')
        if columns:
            title = columns[0].get_text(strip=True)
            link = columns[0].find('a')['href'].strip()
            links.append((title, link))

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_entry_details, url + link, headers): title for title, link in links}
        for future in as_completed(futures):
            title = futures[future]
            contents = future.result()
            catalogue_entries.append({'Title': title, 'Contents': contents})

    for entry in catalogue_entries:
        print(f"Title: {entry['Title']}")
        for title in entry['Contents']:
            print(title)
    return catalogue_entries


