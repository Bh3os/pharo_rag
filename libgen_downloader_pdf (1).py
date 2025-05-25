

import requests
from bs4 import BeautifulSoup
import os
import time
import re
import sys
from urllib.parse import quote_plus, urljoin
import random
from tqdm import tqdm
import concurrent.futures
import threading
from queue import Queue
import logging

# Constants
BASE_URL = "https://libgen.is"
SEARCH_URL = f"{BASE_URL}/search.php"
DOWNLOAD_DIR = "downloaded_books"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]
MAX_WORKERS = 5  # Maximum number of parallel downloads
SEARCH_DELAY = (2, 5)  # Random delay range between searches in seconds

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('libgen_downloader')

# Thread-safe print lock
print_lock = threading.Lock()

# Create download directory if it doesn't exist
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
    logger.info(f"Created directory: {DOWNLOAD_DIR}")

def safe_print(message):
    """Thread-safe print function."""
    with print_lock:
        print(message)

def get_random_user_agent():
    """Return a random user agent to avoid detection."""
    return random.choice(USER_AGENTS)

def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def search_book(title):
    """
    Search for a book on LibGen and return the search results page.
    
    Args:
        title (str): The title of the book to search for
        
    Returns:
        BeautifulSoup object or None: The parsed HTML of the search results page
    """
    safe_print(f"\n[SEARCH] Searching for: {title}")
    
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    params = {
        "req": quote_plus(title),
        "lg_topic": "libgen",
        "open": "0",
        "view": "simple",
        "res": "25",
        "phrase": "1",
        "column": "def",
    }
    
    try:
        safe_print(f"[SEARCH] Sending request to {SEARCH_URL}")
        response = requests.get(SEARCH_URL, params=params, headers=headers, timeout=30)
        
        if response.status_code == 200:
            safe_print(f"[SEARCH] Search successful (Status: {response.status_code})")
            return BeautifulSoup(response.text, "html.parser")
        else:
            safe_print(f"[SEARCH] Search failed with status code: {response.status_code}")
            return None
    except Exception as e:
        safe_print(f"[ERROR] Search error: {str(e)}")
        return None

def find_best_match(soup, title):
    """
    Find the best matching book from search results.
    
    Args:
        soup (BeautifulSoup): The parsed HTML of the search results page
        title (str): The title of the book to match
        
    Returns:
        dict or None: Information about the best matching book
    """
    if not soup:
        return None
    
    safe_print(f"[MATCH] Finding best match for: {title}")
    
    try:
        # Look for the table with search results
        tables = soup.find_all("table")
        if not tables or len(tables) < 3:
            safe_print("[MATCH] No results table found")
            return None
        
        # The results are typically in the third table
        results_table = tables[2]
        rows = results_table.find_all("tr")[1:]  # Skip header row
        
        if not rows:
            safe_print("[MATCH] No results found")
            return None
        
        safe_print(f"[MATCH] Found {len(rows)} potential matches")
        
        # Find the first row with a download link
        for i, row in enumerate(rows):
            cells = row.find_all("td")
            if len(cells) < 10:  # Ensure we have enough cells
                continue
            
            # Extract book information
            book_id = cells[0].text.strip()
            book_title = cells[2].text.strip()
            book_author = cells[1].text.strip()
            book_publisher = cells[3].text.strip()
            book_year = cells[4].text.strip()
            book_pages = cells[5].text.strip()
            book_language = cells[6].text.strip()
            book_size = cells[7].text.strip()
            book_extension = cells[8].text.strip().lower()
            
            # Find download link
            download_cell = cells[9]
            download_links = download_cell.find_all("a")
            
            if not download_links:
                continue
                
            # Get the first download link
            download_url = None
            for link in download_links:
                href = link.get("href")
                if href and (href.startswith("http") or href.startswith("/")):
                    download_url = href if href.startswith("http") else urljoin(BASE_URL, href)
                    break
            
            if download_url:
                safe_print(f"[MATCH] Selected match #{i+1}: {book_title} ({book_author}, {book_year}, {book_extension})")
                return {
                    "id": book_id,
                    "title": book_title,
                    "author": book_author,
                    "publisher": book_publisher,
                    "year": book_year,
                    "pages": book_pages,
                    "language": book_language,
                    "size": book_size,
                    "extension": book_extension,
                    "download_url": download_url,
                }
        
        safe_print("[MATCH] No suitable match found with download link")
        return None
    except Exception as e:
        safe_print(f"[ERROR] Match error: {str(e)}")
        return None

def get_final_download_link(url):
    """
    Follow redirects to get the final download link.
    
    Args:
        url (str): The initial download URL
        
    Returns:
        str or None: The final download URL
    """
    safe_print(f"[DOWNLOAD] Getting final download link from: {url}")
    
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Look for download buttons or links
            download_links = soup.find_all("a", href=True)
            for link in download_links:
                href = link.get("href")
                text = link.text.lower()
                
                # Look for links that seem like download links
                if href and (
                    "download" in text or 
                    "get" in text or 
                    "cloudflare" in href or 
                    "ipfs.io" in href or
                    "gateway" in href
                ):
                    final_url = href if href.startswith("http") else urljoin(url, href)
                    safe_print(f"[DOWNLOAD] Found final download link: {final_url}")
                    return final_url
            
            safe_print("[DOWNLOAD] No final download link found on the page")
            return None
        else:
            safe_print(f"[DOWNLOAD] Failed to get download page with status code: {response.status_code}")
            return None
    except Exception as e:
        safe_print(f"[ERROR] Download link error: {str(e)}")
        return None

def download_book(url, book_info):
    """
    Download a book from the given URL.
    
    Args:
        url (str): The URL to download the book from
        book_info (dict): Information about the book
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    if not url:
        return False
    
    # Create a sanitized filename
    filename = sanitize_filename(f"{book_info['title']} - {book_info['author']} ({book_info['year']}).{book_info['extension']}")
    filepath = os.path.join(DOWNLOAD_DIR, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        safe_print(f"[DOWNLOAD] File already exists: {filepath}")
        return True
    
    safe_print(f"[DOWNLOAD] Downloading: {filename}")
    safe_print(f"[DOWNLOAD] From URL: {url}")
    
    headers = {
        "User-Agent": get_random_user_agent(),
        "Accept": "text/html,application/xhtml+xml,application/xml,application/pdf,application/octet-stream",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    try:
        # Stream the download to show progress
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        
        if response.status_code == 200:
            # Get the total file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Create a progress bar
            progress_bar = tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=f"[DOWNLOAD] {filename}"
            )
            
            # Download the file with progress updates
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            
            # Verify the download
            if os.path.getsize(filepath) > 0:
                safe_print(f"[DOWNLOAD] Successfully downloaded: {filepath}")
                safe_print(f"[DOWNLOAD] File size: {os.path.getsize(filepath) / (1024*1024):.2f} MB")
                return True
            else:
                safe_print(f"[DOWNLOAD] Downloaded file is empty: {filepath}")
                os.remove(filepath)
                return False
        else:
            safe_print(f"[DOWNLOAD] Download failed with status code: {response.status_code}")
            return False
    except Exception as e:
        safe_print(f"[ERROR] Download error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def process_book(title):
    """
    Process a single book: search, find match, and download.
    
    Args:
        title (str): The title of the book to process
        
    Returns:
        bool: True if book was successfully downloaded, False otherwise
    """
    safe_print(f"\n{'='*80}")
    safe_print(f"PROCESSING BOOK: {title}")
    safe_print(f"{'='*80}")
    
    # Step 1: Search for the book
    search_results = search_book(title)
    if not search_results:
        safe_print(f"[PROCESS] Failed to search for book: {title}")
        return False
    
    # Step 2: Find the best match
    book_match = find_best_match(search_results, title)
    if not book_match:
        safe_print(f"[PROCESS] No match found for book: {title}")
        return False
    
    # Step 3: Get the final download link
    final_url = get_final_download_link(book_match["download_url"])
    if not final_url:
        safe_print(f"[PROCESS] Failed to get final download link for book: {title}")
        return False
    
    # Step 4: Download the book
    success = download_book(final_url, book_match)
    if success:
        safe_print(f"[PROCESS] Successfully processed book: {title}")
    else:
        safe_print(f"[PROCESS] Failed to download book: {title}")
    
    return success

def worker(queue, results):
    """
    Worker function for parallel processing.
    
    Args:
        queue (Queue): Queue of book titles to process
        results (dict): Dictionary to store results
    """
    while not queue.empty():
        try:
            # Get a book title from the queue
            index, title = queue.get()
            
            # Process the book
            success = process_book(title)
            
            # Store the result
            results[index] = {
                "title": title,
                "success": success
            }
            
            # Mark the task as done
            queue.task_done()
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            safe_print(f"[ERROR] Worker error: {str(e)}")
            queue.task_done()

def main():
    """Main function to process the list of books in parallel."""
    # List of books to download
    ancient_egypt_books = [
        "The Oxford History of Ancient Egypt",
        "The Complete Gods and Goddesses of Ancient Egypt",
        "The Egyptian Book of the Dead",
        "Temples, Tombs, and Hieroglyphs: A Popular History of Ancient Egypt",
        "The Complete Valley of the Kings",
        "The Complete Tutankhamun",
        "The Complete Pyramids",
        "The Rise and Fall of Ancient Egypt",
        "Chronicle of the Pharaohs",
        "The Search for God in Ancient Egypt",
        "Religion and Ritual in Ancient Egypt",
        "The Mind of Egypt: History and Meaning in the Time of the Pharaohs",
        "The Penguin Historical Atlas of Ancient Egypt",
        "The Pharaohs",
        "The Oxford Essential Guide to Egyptian Mythology",
        "The British Museum Dictionary of Ancient Egypt",
        "The Egyptian World",
        "Daily Life in Ancient Egypt",
        "Egyptian Art",
        "Nefertiti's Face: The Creation of an Icon",
        "Akhenaten: History, Fantasy and Ancient Egypt",
        "The Wisdom of the Egyptians",
        "Magic in Ancient Egypt",
        "The Civilization of Ancient Egypt",
        "The Lost Tomb",
        "The Murder of Tutankhamen",
        "The Heretic Queen",
        "Cleopatra: A Life",
        "Nefertiti: Egypt's Sun Queen",
        "Ramesses: Egypt's Greatest Pharaoh",
        "The Golden Age of the Pharaohs",
        "Egyptian Myth: A Very Short Introduction",
        "Pharaohs and Kings: A Biblical Quest",
        "Egyptian Grammar: Being an Introduction to the Study of Hieroglyphs",
        "The Secret Lore of Egypt",
        "Gods and Myths of Ancient Egypt",
        "The Egyptian Heaven and Hell",
        "The Temple of Man",
        "Egyptian Magic",
        "The World of the Pharaohs",
        "Women in Ancient Egypt",
        "Medicine in Ancient Egypt",
        "The Art of Ancient Egypt",
        "Hieroglyphs: A Very Short Introduction",
        "Egyptian Religion",
        "The History of Ancient Egypt: From the First Farmers to the Great Pyramid",
        "The Egyptian Myths: A Guide to the Ancient Gods and Legends",
        "Tales of Ancient Egypt",
        "The Pyramids: The Mystery, Culture, and Science of Egypt's Great Monuments",
        "Lost Technologies of Ancient Egypt"
    ]
    
    safe_print(f"\n{'*'*80}")
    safe_print(f"LIBGEN BOOK DOWNLOADER (PARALLEL)")
    safe_print(f"{'*'*80}")
    safe_print(f"Starting parallel download of {len(ancient_egypt_books)} books about Ancient Egypt")
    safe_print(f"Using {MAX_WORKERS} parallel workers")
    safe_print(f"Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
    safe_print(f"{'*'*80}\n")
    
    # Create a queue of book titles
    book_queue = Queue()
    for i, title in enumerate(ancient_egypt_books):
        book_queue.put((i, title))
    
    # Dictionary to store results
    results = {}
    
    # Create and start worker threads
    threads = []
    for _ in range(min(MAX_WORKERS, len(ancient_egypt_books))):
        thread = threading.Thread(target=worker, args=(book_queue, results))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Wait for all tasks to be processed
    book_queue.join()
    
    # Print summary
    successful = sum(1 for result in results.values() if result["success"])
    failed = len(results) - successful
    
    safe_print(f"\n{'*'*80}")
    safe_print(f"DOWNLOAD SUMMARY")
    safe_print(f"{'*'*80}")
    safe_print(f"Total books: {len(ancient_egypt_books)}")
    safe_print(f"Successfully downloaded: {successful}")
    safe_print(f"Failed to download: {failed}")
    safe_print(f"Success rate: {successful/len(ancient_egypt_books)*100:.2f}%")
    safe_print(f"Download directory: {os.path.abspath(DOWNLOAD_DIR)}")
    safe_print(f"{'*'*80}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        safe_print("\n[MAIN] Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        safe_print(f"\n[ERROR] Unexpected error: {str(e)}")
        sys.exit(1)
