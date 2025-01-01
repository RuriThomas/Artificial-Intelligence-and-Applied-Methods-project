import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import hashlib
import os
import csv
url = 'https://www.british-history.ac.uk'
csv.field_size_limit(2147483647)

def is_table_of_contents(soup):
    h2_headers = soup.find_all('h2')
    for header in h2_headers:
        if header.get_text(strip=True) == "Table of Contents":
            return True
    return False

def check_pagination(base_url, headers=None):
    page = 0
    has_more_pages = True
    links = [] 
    
    while has_more_pages:
        url = f"{base_url}?page={page}"
        print(f"Checking: {url}")
        
        response = requests.get(url, headers=headers)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        rows = soup.select('table:first-of-type tr')
        if rows!=None:
            for row in rows:
                columns = row.find_all('td')
                if columns:
                    link = columns[0].find('a')['href'].strip()
                    links.append(link)
            next_button = soup.find('a', {'class': 'page-link', 'rel': 'next', 'title': 'Go to next page'}) 
            if not next_button:
                print("No next button found.")
                has_more_pages = False
        else:
            has_more_pages = False
        page += 1

    return links

def isDirectory(directory,soup):
    return "/series/" in directory or "catalogue" in directory or is_table_of_contents(soup)


def getLocal(path):
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        return [row for row in reader]

def saveLocal(dir,data):
    full_url = url + dir
    storage_dir = "dataset"
    hashed_filename = hashlib.sha256(full_url.encode()).hexdigest()
    path_name = os.path.join(storage_dir,hashed_filename)
    if os.path.exists(path_name):
        print(f"Data for {full_url} already stored at {path_name}")
    
    with open(path_name, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, ["Title", "Content"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(data)

def load_data(headers, directory):
    full_url = url + directory
    storage_dir = "dataset"
    hashed_filename = hashlib.sha256(full_url.encode()).hexdigest()
    path_name = os.path.join(storage_dir,hashed_filename)
    if os.path.exists(path_name):
        print(f"Data for {full_url} already stored at {path_name}")
        return getLocal(path_name)
    print(f"Fetching data at {full_url}")
    

    response = requests.get(full_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    catalogue_entries = []
    links = []

    # If the current directory is an article, process it directly
    if not isDirectory(directory,soup):
        title_elem = soup.find('div', {'id': 'block-bho-page-title'})
        content_elem = soup.find('div', {'class': 'inner'})
        if title_elem and content_elem:
            title = title_elem.get_text(strip=True)
            paragraphs = content_elem.find_all('p')
            content = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
            clean_content = re.sub(r'\s+', ' ', content).strip()
            return [{'Title': title, 'Content': clean_content}]
        
    # If it is a search page, save links to volume(s)
    if "/series/" in directory:
        results = soup.find_all('div', class_='views-field views-field-title')
        for result in results:
            title_tag = result.find('a')
            link = title_tag['href']
            links.append(link)
    else:
        # If it is table of contents save links in all pages
        if is_table_of_contents(soup):
            links.extend(check_pagination(full_url,headers))
        else:
            rows = soup.select('table:first-of-type tr')
            for row in rows:
                columns = row.find_all('td')
                if columns:
                    link = columns[0].find('a')['href'].strip()
                    links.append(link)

    # Use multithreading to process each link concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(load_data, headers, link): link for link in links}
        for future in as_completed(futures):
            try:
                entries = future.result()
                if entries:
                    catalogue_entries.extend(entries)
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")
    return catalogue_entries
    
    


