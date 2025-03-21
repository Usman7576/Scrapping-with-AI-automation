import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import threading

# Base URL for NeurIPS proceedings
BASE_URL = "https://papers.nips.cc"

# Years to scrape (2020 to 2024)
YEARS = range(2020, 2025)

# Folder to save PDFs
PDF_FOLDER = "neurips_pdfs"
if not os.path.exists(PDF_FOLDER):
    print(f"Creating folder: {PDF_FOLDER}")
    os.makedirs(PDF_FOLDER)

# CSV file to store metadata
CSV_FILE = "neurips_2020_2024_data.csv"

# Thread-safe list to store all paper data
all_papers = []
lock = threading.Lock()

# Initialize CSV with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["title", "authors", "abstract", "pdf_url", "year"])
    df.to_csv(CSV_FILE, index=False)
    print(f"Created CSV file: {CSV_FILE}")

def get_paper_links(year):
    """Get all paper page links for a given year."""
    url = f"{BASE_URL}/paper_files/paper/{year}"
    print(f"Fetching URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=True)
        # Look for paper links
        paper_links = []
        for link in links:
            href = link["href"]
            # Look for links starting with /paper_files/paper/{year}/ but exclude /file/ links
            if href.startswith(f"/paper_files/paper/{year}/") and "/file/" not in href:
                full_url = f"{BASE_URL}{href}"
                if full_url not in paper_links:  # Avoid duplicates
                    paper_links.append(full_url)
                    print(f"Found paper link: {full_url}")
        print(f"Found {len(paper_links)} papers for {year}")
        return paper_links
    except Exception as e:
        print(f"Error fetching links for {year}: {e}")
        return []

def scrape_paper_data(url):
    """Scrape title, authors, abstract, and PDF link from a paper page."""
    print(f"Scraping URL: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract data
        title = soup.find("h4").text.strip() if soup.find("h4") else "N/A"
        # Authors are in a <p> tag with class="authors"
        author_p = soup.find("p", class_="authors")
        authors = author_p.text.strip() if author_p else "N/A"
        # Abstract is in a <p> tag with class="abstract"
        abstract_p = soup.find("p", class_="abstract")
        abstract = abstract_p.text.strip() if abstract_p else "N/A"
        
        # Find PDF link (try multiple labels)
        pdf_link = None
        for label in ["Download PDF", "PDF", "Paper PDF"]:
            pdf_link = soup.find("a", href=True, string=label)
            if pdf_link:
                break
        if not pdf_link:
            # Fallback: look for any link ending in -Paper.pdf
            for link in soup.find_all("a", href=True):
                if link["href"].endswith("-Paper.pdf"):
                    pdf_link = link
                    break
        pdf_url = f"{BASE_URL}{pdf_link['href']}" if pdf_link else None
        
        print(f"Title: {title}, PDF URL: {pdf_url}")
        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "pdf_url": pdf_url,
            "year": url.split("/")[5]  # Extract year from URL
        }
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def download_pdf(paper_data):
    """Download PDF file and save it with a sanitized filename."""
    if not paper_data or not paper_data["pdf_url"]:
        print(f"No PDF URL for paper: {paper_data}")
        return
    title = paper_data["title"]
    year = paper_data["year"]
    pdf_url = paper_data["pdf_url"]
    
    # Sanitize title for filename
    safe_title = "".join(c for c in title if c.isalnum() or c in " -_").strip()[:50]
    filename = f"{PDF_FOLDER}/{year}_{safe_title}.pdf"
    
    print(f"Attempting to download: {pdf_url} to {filename}")
    if not os.path.exists(filename):
        try:
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded: {filename}")
        except Exception as e:
            print(f"Error downloading {pdf_url}: {e}")
    else:
        print(f"Skipped (already exists): {filename}")

def append_to_csv(paper_data):
    """Append paper data to the CSV file in a thread-safe manner."""
    with lock:
        df = pd.DataFrame([paper_data])
        df.to_csv(CSV_FILE, mode="a", header=False, index=False)
        print(f"Appended data to CSV: {paper_data['title']}")

def process_paper(url):
    """Scrape a single paper, download its PDF, and append to CSV."""
    paper_data = scrape_paper_data(url)
    if paper_data:
        download_pdf(paper_data)
        append_to_csv(paper_data)
    return paper_data

def process_year(year):
    """Process all papers for a given year using multithreading."""
    print(f"\nProcessing year {year}...")
    paper_links = get_paper_links(year)
    
    if not paper_links:
        print(f"No papers found for {year}, skipping...")
        return
    
    # Scrape and download concurrently
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(process_paper, url): url for url in paper_links}
        for future in tqdm(as_completed(future_to_url), total=len(paper_links), desc=f"Processing {year}"):
            paper_data = future.result()
            if paper_data:
                with lock:
                    all_papers.append(paper_data)
            time.sleep(0.2)  # Small delay per thread to avoid overwhelming the server

def main():
    # Process each year sequentially
    for year in YEARS:
        process_year(year)
    
    # Final save to CSV (optional, since we're appending as we go)
    if all_papers:
        print(f"Final save to CSV: {CSV_FILE}")
        df = pd.DataFrame(all_papers)
        df.to_csv(CSV_FILE, index=False)
        print(f"Saved data to {CSV_FILE}")
    else:
        print("No data to save to CSV")

if __name__ == "__main__":
    main()