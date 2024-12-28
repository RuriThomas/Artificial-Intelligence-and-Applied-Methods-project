import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

url = 'https://www.british-history.ac.uk'

def has_table_of_contents(soup):
    h2_headers = soup.find_all('h2')
    for header in h2_headers:
        if header.get_text(strip=True) == "Table of Contents":
            return True
    return False
def isDirectory(directory,soup):
    return "/series/" in directory or "catalogue" in directory or has_table_of_contents(soup)

def load_data(headers, directory):
    full_url = url + directory
    response = requests.get(full_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    catalogue_entries = []

    # If the current directory is an article, process it directly
    if not isDirectory(directory,soup):
        print("Found an article")
        title_elem = soup.find('div', {'id': 'block-bho-page-title'})
        content_elem = soup.find('div', {'class': 'inner'})
        if title_elem and content_elem:
            title = title_elem.get_text(strip=True)
            content = content_elem.get_text(strip=True)
            return [{'Title': title, 'Content': content}]
    # Otherwise, parse the directory for links to entries or sub-directories
    if "/series/" in directory:
        print("")
    else:
        rows = soup.select('table:first-of-type tr')
        links = []
        print("Found a directory")
        for row in rows:
            columns = row.find_all('td')
            if columns:
                link = columns[0].find('a')['href'].strip()
                print(link)
                links.append(link)

    # Use multithreading to process each link concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(load_data, headers, link): link for link in links}
        for future in as_completed(futures):
            try:
                entries = future.result()
                if entries:
                    catalogue_entries.extend(entries)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")
    
    return catalogue_entries
    
    


