import requests
from bs4 import BeautifulSoup
import csv
import time
import random
from datetime import datetime
import re
from urllib.parse import urljoin, urlparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp

class IndonesianNewsScraper:
    def __init__(self):
        self.articles = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.max_workers = 12  # Increased workers
        self.timeout = 45  # Increased timeout
        
        # Define categories and their URLs for different news sources - fixed broken URLs
        self.categories = {
            'liputan6': {
                'olahraga_non_bola': 'https://www.liputan6.com/olahraga',
                'liga_inggris': 'https://www.liputan6.com/bola/liga-inggris',
                'liga_indonesia': 'https://www.liputan6.com/bola/liga-indonesia',
                'liga_spanyol': 'https://www.liputan6.com/bola/liga-spanyol',
                'liga_italia': 'https://www.liputan6.com/bola/liga-italia'
            },
            'detik': {
                'olahraga_non_bola': 'https://sport.detik.com/indeks',
                'liga_inggris': 'https://sport.detik.com/sepakbola/liga-inggris',
                'liga_indonesia': 'https://sport.detik.com/sepakbola/liga-indonesia',
                'liga_spanyol': 'https://sport.detik.com/sepakbola/liga-spanyol',
                'liga_italia': 'https://sport.detik.com/sepakbola/liga-italia'
            },
            'kompas': {
                'olahraga_non_bola': 'https://bola.kompas.com/olahraga',
                'liga_inggris': 'https://bola.kompas.com/liga-inggris',
                'liga_indonesia': 'https://bola.kompas.com/liga-indonesia',
                'liga_spanyol': 'https://bola.kompas.com/liga-spanyol',
                'liga_italia': 'https://bola.kompas.com/liga-italia'
            },
            'tribun': {
                'olahraga_non_bola': 'https://www.tribunnews.com/sport',
                'liga_inggris': 'https://www.tribunnews.com/sepakbola/liga-inggris',
                'liga_indonesia': 'https://www.tribunnews.com/sepakbola/liga-indonesia',
                # Fixed broken league URLs for Tribun - fallback to general sport
                'liga_spanyol': 'https://www.tribunnews.com/sport',
                'liga_italia': 'https://www.tribunnews.com/sport'
            },
            'cnn': {
                'olahraga_non_bola': 'https://www.cnnindonesia.com/olahraga',
                'liga_inggris': 'https://www.cnnindonesia.com/olahraga/sepakbola',
                'liga_indonesia': 'https://www.cnnindonesia.com/olahraga/sepakbola',
                'liga_spanyol': 'https://www.cnnindonesia.com/olahraga/sepakbola',
                'liga_italia': 'https://www.cnnindonesia.com/olahraga/sepakbola'
            }
        }
    
    def scrape_liputan6(self, category, max_articles=60):
        """Scrape articles from Liputan6"""
        articles = []
        base_url = self.categories['liputan6'][category]
        
        for page in range(1, 25):  # Increased page range
            try:
                url = f"{base_url}?page={page}"
                print(f"Scraping Liputan6 - {category} - Page {page}")
                
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links - try multiple possible selectors
                article_links = []
                
                # Try different possible selectors for Liputan6
                selectors = [
                    'a.articles--iridescent-list--text-item__title-link',
                    'a[href*="/news/"]',
                    'a[href*="/read/"]',
                    'h3 a[href]',
                    'h2 a[href]',
                    '.media__text a',
                    '.article__title a',
                    'a[href*="/berita/"]',
                    'a[href*="/bola/"]'
                ]
                
                for selector in selectors:
                    links = soup.select(selector)
                    if links:
                        article_links.extend(links[:20])  # Get more links per page
                        break
                
                # Remove duplicates while preserving order
                seen_urls = set()
                unique_links = []
                for link in article_links:
                    href = link.get('href', '')
                    if href and href not in seen_urls:
                        seen_urls.add(href)
                        unique_links.append(link)
                
                article_links = unique_links[:18]  # Increased limit per page
                
                for link in article_links:
                    try:
                        href = link.get('href', '')
                        if not href:
                            continue
                            
                        article_url = urljoin('https://www.liputan6.com', href)
                        title = link.get_text(strip=True)
                        
                        if not title or len(title) < 10:  # Skip if title is too short
                            continue
                        
                        # Get article content
                        content = self.get_liputan6_article_content(article_url)
                        
                        if content and len(content) > 150:  # Only save if content is substantial
                            articles.append({
                                'title': title,
                                'content': content,
                                'url': article_url,
                                'source': 'Liputan6',
                                'category': category,
                                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
                            if len(articles) >= max_articles:
                                break
                        
                        time.sleep(random.uniform(1, 2))  # Be respectful with requests
                        
                    except Exception as e:
                        print(f"Error processing article link: {e}")
                        continue
                
                if len(articles) >= max_articles:
                    break
                    
            except Exception as e:
                print(f"Error scraping Liputan6 page {page}: {e}")
                continue
                
        return articles
    
    def scrape_tribun(self, category, max_articles=25):
        """Scrape articles from Tribun News"""
        articles = []
        base_url = self.categories['tribun'][category]
        
        # Try different URL patterns for pagination
        page_patterns = [
            lambda url, page: url if page == 1 else f"{url}/page/{page}",
            lambda url, page: url if page == 1 else f"{url}?page={page}",
            lambda url, page: url if page == 1 else f"{url}/index{page}.html",
            lambda url, page: url if page == 1 else f"{url}&page={page}"
        ]
        
        for page in range(1, 8):  # Try up to 7 pages
            success = False
            
            for pattern in page_patterns:
                try:
                    url = pattern(base_url, page)
                    print(f"Trying Tribun - {category} - Page {page}: {url}")
                    
                    response = requests.get(url, headers=self.headers, timeout=15)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find article links - try multiple selectors for Tribun
                    article_links = []
                    
                    # Try different possible selectors for Tribun
                    selectors = [
                        'a.f16',
                        'a[href*="/news/"]',
                        'h3 a[href]',
                        'h2 a[href]',
                        '.lsi a[href]',
                        '.ptb15 a[href]',
                        '.fr a[href]',
                        'article a[href]',
                        '.list-berita a[href]'
                    ]
                    
                    for selector in selectors:
                        links = soup.select(selector)
                        if links:
                            article_links.extend(links[:15])
                            break
                    
                    # Remove duplicates while preserving order
                    seen_urls = set()
                    unique_links = []
                    for link in article_links:
                        href = link.get('href', '')
                        if href and href not in seen_urls:
                            seen_urls.add(href)
                            unique_links.append(link)
                    
                    article_links = unique_links[:12]
                    
                    if article_links:  # If we found articles, this pattern works
                        success = True
                        
                        for link in article_links:
                            try:
                                href = link.get('href', '')
                                if not href:
                                    continue
                                    
                                article_url = urljoin('https://www.tribunnews.com', href)
                                title = link.get_text(strip=True)
                                
                                if not title or len(title) < 10:  # Skip if title is too short
                                    continue
                                
                                # Get article content
                                content = self.get_tribun_article_content(article_url)
                                
                                if content and len(content) > 150:  # Only save if content is substantial
                                    articles.append({
                                        'title': title,
                                        'content': content,
                                        'url': article_url,
                                        'source': 'Tribun',
                                        'category': category,
                                        'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                    })
                                    
                                    if len(articles) >= max_articles:
                                        break
                                
                                time.sleep(random.uniform(1, 2))  # Be respectful with requests
                                
                            except Exception as e:
                                print(f"Error processing article link: {e}")
                                continue
                        
                        break  # Found working pattern, move to next page
                            
                except Exception as e:
                    print(f"  Pattern failed: {e}")
                    continue
            
            if not success:
                print(f"All patterns failed for page {page}")
                break
            
            if len(articles) >= max_articles:
                break
                
        return articles
    
    def get_tribun_article_content(self, url):
        """Get full article content from Tribun"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find article content - Tribun uses different structure
            content_div = soup.find('div', class_='side-article txt-article')
            if not content_div:
                content_div = soup.find('div', class_='txt-article')
            
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                return content
            
            return ""
            
        except Exception as e:
            print(f"Error getting article content from {url}: {e}")
            return ""
    
    def scrape_kompas(self, category, max_articles=25):
        """Scrape articles from Kompas"""
        articles = []
        base_url = self.categories['kompas'][category]
        
        for page in range(1, 8):  # Try up to 7 pages
            try:
                url = f"{base_url}?page={page}"
                print(f"Scraping Kompas - {category} - Page {page}")
                
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find article links - try multiple selectors for Kompas
                article_links = []
                
                # Try different possible selectors for Kompas
                selectors = [
                    'a.article__link',
                    'a[href*="/read/"]',
                    'h3 a[href]',
                    'h2 a[href]',
                    'a[href*="/news/"]'
                ]
                
                for selector in selectors:
                    links = soup.select(selector)
                    if links:
                        article_links.extend(links[:15])
                        break
                
                # Remove duplicates while preserving order
                seen_urls = set()
                unique_links = []
                for link in article_links:
                    href = link.get('href', '')
                    if href and href not in seen_urls:
                        seen_urls.add(href)
                        unique_links.append(link)
                
                article_links = unique_links[:12]
                
                for link in article_links:
                    try:
                        href = link.get('href', '')
                        if not href:
                            continue
                            
                        article_url = urljoin('https://sport.kompas.com', href)
                        title = link.get_text(strip=True)
                        
                        if not title or len(title) < 10:  # Skip if title is too short
                            continue
                        
                        # Get article content
                        content = self.get_kompas_article_content(article_url)
                        
                        if content and len(content) > 150:  # Only save if content is substantial
                            articles.append({
                                'title': title,
                                'content': content,
                                'url': article_url,
                                'source': 'Kompas',
                                'category': category,
                                'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
                            if len(articles) >= max_articles:
                                break
                        
                        time.sleep(random.uniform(1, 2))  # Be respectful with requests
                        
                    except Exception as e:
                        print(f"Error processing article link: {e}")
                        continue
                
                if len(articles) >= max_articles:
                    break
                    
            except Exception as e:
                print(f"Error scraping Kompas page {page}: {e}")
                continue
                
        return articles
    
    def get_kompas_article_content(self, url):
        """Get full article content from Kompas"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find title
            title = ""
            title_elem = soup.find('h1', class_='read__title')
            if not title_elem:
                title_elem = soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Find article content - Kompas uses different structure
            content_div = soup.find('div', class_='read__content')
            if not content_div:
                content_div = soup.find('div', class_='article__content')
            
            content = ""
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            return {
                'title': title,
                'content': content
            }
            
        except Exception as e:
            print(f"Error getting article content from {url}: {e}")
            return None
    
    def get_liputan6_article_content(self, url):
        """Get full article content from Liputan6"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find title
            title = ""
            title_elem = soup.find('h1', class_='article-header__title')
            if not title_elem:
                title_elem = soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Find article content
            content_div = soup.find('div', class_='article-content-body__item-content')
            if not content_div:
                content_div = soup.find('div', class_='read-page--content')
            
            content = ""
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            return {
                'title': title,
                'content': content
            }
            
        except Exception as e:
            print(f"Error getting article content from {url}: {e}")
            return None
    
    def scrape_detik(self, category, max_articles=50):
        """Scrape articles from Detik"""
        articles = []
        base_url = self.categories['detik'][category]
        
        for page in range(1, 25):  # Increased page range
            try:
                print(f"Scraping Detik - {category} - Page {page}")
                
                # Construct page URL
                if page == 1:
                    url = base_url
                else:
                    url = f"{base_url}?page={page}"
                
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Multiple selectors for article links
                article_links = []
                selectors = [
                    'h3 a[href*="/sepakbola/"]',
                    'h2 a[href*="/sepakbola/"]',
                    'a[href*="/sepakbola/"]',
                    '.media__title a',
                    '.media__text a',
                    'article a',
                    '.list__item a',
                    '.media a',
                    'a[href*="/sport/"]',
                    '.media__title a',
                    '.list_media__title a',
                    'h2 a',
                    'h3 a',
                    '.title a'
                ]
                
                for selector in selectors:
                    links = soup.select(selector)
                    for link in links:
                        href = link.get('href', '')
                        if href and href.startswith('http') and ('/sepakbola/' in href or '/sport/' in href):
                            article_links.append(href)
                
                # Remove duplicates
                article_links = list(set(article_links))
                
                for link in article_links:
                    if len(articles) >= max_articles:
                        break
                    
                    try:
                        article_data = self.get_detik_article_content(link)
                        if article_data and len(article_data['content']) > 150 and len(article_data['title']) > 10:
                            article_data['source'] = 'Detik'
                            article_data['category'] = category
                            articles.append(article_data)
                            print(f"  - {article_data['title'][:60]}...")
                            
                    except Exception as e:
                        print(f"  Error processing article {link}: {e}")
                        continue
                
                if len(articles) >= max_articles:
                    break
                    
            except Exception as e:
                print(f"Error scraping Detik page {page}: {e}")
                continue
        
        return articles
    
    def get_detik_article_content(self, url):
        """Get full article content from Detik"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find title
            title = ""
            title_elem = soup.find('h1', class_='detail__title')
            if not title_elem:
                title_elem = soup.find('h1')
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Find article content - Detik uses different structure
            content_div = soup.find('div', class_='detail__body')
            if not content_div:
                content_div = soup.find('div', class_='itp_bodycontent')
            
            content = ""
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            
            return {
                'title': title,
                'content': content
            }
            
        except Exception as e:
            print(f"Error getting article content from {url}: {e}")
            return None
    
    def scrape_all_categories(self, target_per_category=20):
        """Scrape all categories from all sources"""
        all_articles = []
        
        # Scrape Liputan6
        print("\n=== SCRAPING LIPUTAN6 ===")
        for category in self.categories['liputan6'].keys():
            print(f"\nScraping category: {category}")
            articles = self.scrape_liputan6(category, target_per_category)
            all_articles.extend(articles)
            print(f"Collected {len(articles)} articles from Liputan6 - {category}")
        
        # Scrape Detik
        print("\n=== SCRAPING DETIK ===")
        for category in self.categories['detik'].keys():
            print(f"\nScraping category: {category}")
            articles = self.scrape_detik(category, target_per_category)
            all_articles.extend(articles)
            print(f"Collected {len(articles)} articles from Detik - {category}")
        
        # Scrape Kompas
        print("\n=== SCRAPING KOMPAS ===")
        for category in self.categories['kompas'].keys():
            print(f"\nScraping category: {category}")
            articles = self.scrape_kompas(category, target_per_category)
            all_articles.extend(articles)
            print(f"Collected {len(articles)} articles from Kompas - {category}")
        
        # Scrape Tribun
        print("\n=== SCRAPING TRIBUN ===")
        for category in self.categories['tribun'].keys():
            print(f"\nScraping category: {category}")
            articles = self.scrape_tribun(category, target_per_category)
            all_articles.extend(articles)
            print(f"Collected {len(articles)} articles from Tribun - {category}")
        
        # Scrape CNN Indonesia
        print("\n=== SCRAPING CNN INDONESIA ===")
        for category in self.categories['cnn'].keys():
            print(f"\nScraping category: {category}")
            articles = self.scrape_cnn(category, target_per_category)
            all_articles.extend(articles)
            print(f"Collected {len(articles)} articles from CNN - {category}")
        
        return all_articles
    
    def scrape_category_concurrent(self, source, category, max_articles=25):
        """Scrape a specific category using concurrent execution"""
        articles = []
        
        if source == 'liputan6':
            articles = self.scrape_liputan6(category, max_articles)
        elif source == 'detik':
            articles = self.scrape_detik(category, max_articles)
        elif source == 'tribun':
            articles = self.scrape_tribun(category, max_articles)
        
        return {'source': source, 'category': category, 'articles': articles}
    
    def run_async_multithread(self, min_articles=100):
        """Run asynchronous multi-threaded scraping - Optimized version"""
        print("🚀 Starting Optimized Asynchronous Multi-threaded Indonesian News Scraping...")
        print(f"Target: {min_articles} articles from multiple sources")
        
        all_articles = []
        tasks = []
        max_workers = 12  # Increased concurrent threads
        articles_per_task = min_articles // 8  # Better workload distribution
        
        print(f"\n⚡ Starting {max_workers} concurrent workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Liputan6 tasks - increased limits
            print("\n=== SCHEDULING LIPUTAN6 TASKS ===")
            for category in self.categories['liputan6'].keys():
                task = executor.submit(self.scrape_category_concurrent, 'liputan6', category, articles_per_task + 8)
                tasks.append(task)
                print(f"Scheduled: Liputan6 - {category}")
            
            # Detik tasks - increased limits
            print("\n=== SCHEDULING DETIK TASKS ===")
            for category in self.categories['detik'].keys():
                task = executor.submit(self.scrape_category_concurrent, 'detik', category, articles_per_task + 8)
                tasks.append(task)
                print(f"Scheduled: Detik - {category}")
            
            # Tribun tasks - increased limits
            print("\n=== SCHEDULING TRIBUN TASKS ===")
            for category in self.categories['tribun'].keys():
                task = executor.submit(self.scrape_category_concurrent, 'tribun', category, articles_per_task + 6)
                tasks.append(task)
                print(f"Scheduled: Tribun - {category}")
            
            # Kompas tasks - added back with increased limits
            print("\n=== SCHEDULING KOMPAS TASKS ===")
            for category in self.categories['kompas'].keys():
                task = executor.submit(self.scrape_category_concurrent, 'kompas', category, articles_per_task + 6)
                tasks.append(task)
                print(f"Scheduled: Kompas - {category}")
            
            # CNN tasks - added with increased limits
            print("\n=== SCHEDULING CNN TASKS ===")
            for category in self.categories['cnn'].keys():
                task = executor.submit(self.scrape_category_concurrent, 'cnn', category, articles_per_task + 5)
                tasks.append(task)
                print(f"Scheduled: CNN - {category}")
            
            # Collect results with better progress tracking
            completed = 0
            total_tasks = len(tasks)
            target_reached = False
            print("\n=== COLLECTING RESULTS ===")
            
            for future in as_completed(tasks):
                completed += 1
                try:
                    result = future.result()
                    all_articles.extend(result['articles'])
                    print(f"✅ [{completed}/{total_tasks}] {result['source']} - {result['category']} ({len(result['articles'])} articles)")
                    
                    # Show progress
                    print(f"📈 Progress: {len(all_articles)}/{min_articles} articles ({len(all_articles)/min_articles*100:.1f}%)")
                    
                    # Early exit if we have enough articles
                    if len(all_articles) >= min_articles and not target_reached:
                        print(f"🎯 Target reached! Collected {len(all_articles)} articles.")
                        target_reached = True
                        # Cancel remaining tasks
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        break
                        
                except Exception as e:
                    print(f"❌ [{completed}/{total_tasks}] Task failed: {e}")
            
            # If target was reached, save and exit immediately
            if target_reached:
                print(f"✅ Target reached! Collected {len(all_articles)} out of {min_articles}")
                self.save_to_csv(all_articles)
                print(f"✅ Successfully saved {len(all_articles)} articles to news_articles.csv")
                self.print_summary(all_articles)
                print(f"\n🏁 Optimized asynchronous multi-threaded scraping completed! Total articles: {len(all_articles)}")
                return all_articles
        
        print(f"\n📈 Total articles collected: {len(all_articles)}")
        
        # Final check: collect any remaining results from completed tasks
        print("🔍 Final check for any remaining articles...")
        for task in tasks:
            if task.done() and not task.cancelled():
                try:
                    result = task.result()
                    # Only add if this result wasn't already processed
                    if result and 'articles' in result:
                        existing_urls = {article['url'] for article in all_articles}
                        new_articles = [article for article in result['articles'] if article['url'] not in existing_urls]
                        if new_articles:
                            all_articles.extend(new_articles)
                            print(f"Added {len(new_articles)} additional articles from completed task")
                except Exception as e:
                    print(f"Error processing final task result: {e}")
        
        print(f"📈 Final total articles collected: {len(all_articles)}")
        
        if len(all_articles) >= min_articles:
            print(f"✅ Target reached! Collected {len(all_articles)} out of {min_articles}")
            self.save_to_csv(all_articles)
            print(f"✅ Successfully saved {len(all_articles)} articles to news_articles.csv")
        else:
            print(f"⚠️  Below target. Collected {len(all_articles)} out of {min_articles}")
            if all_articles:
                self.save_to_csv(all_articles)
                print(f"✅ Successfully saved {len(all_articles)} articles to news_articles.csv")
            else:
                print("❌ No articles to save!")
        
        # Print summary
        self.print_summary(all_articles)
        
        print(f"\n🏁 Optimized asynchronous multi-threaded scraping completed! Total articles: {len(all_articles)}")
        return all_articles
    
    def scrape_cnn(self, category, max_articles=20):
        """Scrape articles from CNN Indonesia"""
        articles = []
        base_url = self.categories['cnn'][category]
        
        # Try different URL patterns for pagination
        page_patterns = [
            lambda url, page: url if page == 1 else f"{url}/page/{page}",
            lambda url, page: url if page == 1 else f"{url}?page={page}",
            lambda url, page: url if page == 1 else f"{url}&page={page}"
        ]
        
        for page in range(1, 8):  # Try up to 7 pages
            success = False
            
            for pattern in page_patterns:
                try:
                    url = pattern(base_url, page)
                    print(f"Trying CNN - {category} - Page {page}: {url}")
                    
                    response = requests.get(url, headers=self.headers, timeout=15)
                    response.raise_for_status()
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Multiple selectors for CNN Indonesia article links
                    article_links = []
                    selectors = [
                        'h3 a[href*="/news/"]',
                        'h2 a[href*="/news/"]',
                        'a[href*="/news/"]',
                        '.media_rows a',
                        '.list_media a',
                        'article a',
                        '.box_text a'
                    ]
                    
                    for selector in selectors:
                        links = soup.select(selector)
                        for link in links:
                            href = link.get('href', '')
                            if href and href.startswith('http') and '/news/' in href:
                                article_links.append(href)
                    
                    # Remove duplicates
                    article_links = list(set(article_links))
                    
                    if article_links:  # If we found articles, this pattern works
                        success = True
                        
                        for link in article_links:
                            if len(articles) >= max_articles:
                                break
                            
                            try:
                                article_data = self.get_cnn_article_content(link)
                                if article_data and len(article_data['content']) > 150 and len(article_data['title']) > 10:
                                    article_data['source'] = 'CNN Indonesia'
                                    article_data['category'] = category
                                    articles.append(article_data)
                                    print(f"  - {article_data['title'][:60]}...")
                                    
                            except Exception as e:
                                print(f"  Error processing article {link}: {e}")
                                continue
                        
                        break  # Found working pattern, move to next page
                        
                except Exception as e:
                    print(f"  Pattern failed: {e}")
                    continue
            
            if not success:
                print(f"All patterns failed for page {page}")
                break
            
            if len(articles) >= max_articles:
                break
        
        return articles
    
    def get_cnn_article_content(self, url):
        """Extract content from CNN Indonesia article"""
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = ''
            title_selectors = ['h1', '.title', 'h1[itemprop="headline"]']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract content
            content = ''
            content_selectors = [
                '.detail_text',
                '.text_detail',
                'div[itemprop="articleBody"]',
                '.content_detail'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove scripts, styles, and unwanted elements
                    for script in content_elem(['script', 'style', 'div.sembunyikan']):
                        script.decompose()
                    
                    content = content_elem.get_text(separator=' ', strip=True)
                    break
            
            if title and content:
                return {
                    'title': title,
                    'content': content,
                    'url': url,
                    'scraped_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            
        except Exception as e:
            print(f"Error extracting CNN content: {e}")
            return None
        
        return None

    def save_to_csv(self, articles=None, filename='news_articles.csv'):
        """Save scraped articles to CSV file"""
        if articles is None:
            articles = self.articles
            
        if not articles:
            print("No articles to save!")
            return
        
        # Define CSV columns
        fieldnames = ['title', 'content', 'url', 'source', 'category', 'scraped_at']
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(articles)
            
            print(f"\n✅ Successfully saved {len(articles)} articles to {filename}")
            
            # Print summary
            sources = {}
            categories = {}
            
            for article in articles:
                source = article['source']
                category = article['category']
                sources[source] = sources.get(source, 0) + 1
                categories[category] = categories.get(category, 0) + 1
            
            print("\n📊 SUMMARY:")
            print("By Source:")
            for source, count in sources.items():
                print(f"  - {source}: {count} articles")
            
            print("\nBy Category:")
            for category, count in categories.items():
                print(f"  - {category}: {count} articles")
                
        except Exception as e:
            print(f"Error saving to CSV: {e}")
    
    def run(self, min_articles=100):
        """Run the complete scraping process"""
        print("🚀 Starting Indonesian News Scraping...")
        print(f"Target: {min_articles} articles from multiple sources")
        
        # Scrape all categories
        articles = self.scrape_all_categories()
        
        print(f"\n📈 Total articles collected: {len(articles)}")
        
        if len(articles) >= min_articles:
            print(f"✅ Target reached! Collected {len(articles)} out of {min_articles}")
            self.save_to_csv(articles)
            print(f"✅ Successfully saved {len(articles)} articles to news_articles.csv")
        else:
            print(f"⚠️  Below target. Collected {len(articles)} out of {min_articles}")
            if articles:
                self.save_to_csv(articles)
                print(f"✅ Successfully saved {len(articles)} articles to news_articles.csv")
            else:
                print("❌ No articles to save!")
        
        # Print summary
        self.print_summary(articles)
        
        print(f"\n🏁 Scraping completed! Total articles: {len(articles)}")
        return articles
    
    def run_focused(self, min_articles=100):
        """Run focused scraping on most reliable sources"""
        print("🚀 Starting Focused Indonesian News Scraping...")
        print(f"Target: {min_articles} articles from reliable sources")
        
        all_articles = []
        target_per_source = min_articles // 2  # Split between 2 main sources
        
        # Focus on most reliable sources first
        print("\n=== FOCUSING ON RELIABLE SOURCES ===")
        
        # Liputan6 - most reliable
        print("\n=== SCRAPING LIPUTAN6 (Primary Source) ===")
        for category in self.categories['liputan6'].keys():
            print(f"\nScraping category: {category}")
            articles = self.scrape_liputan6(category, target_per_source // 5 + 5)  # More per category
            all_articles.extend(articles)
            print(f"Collected {len(articles)} articles from Liputan6 - {category}")
        
        # Detik - second most reliable
        print("\n=== SCRAPING DETIK (Secondary Source) ===")
        for category in self.categories['detik'].keys():
            print(f"\nScraping category: {category}")
            articles = self.scrape_detik(category, target_per_source // 5 + 5)  # More per category
            all_articles.extend(articles)
            print(f"Collected {len(articles)} articles from Detik - {category}")
        
        # Add Tribun and Kompas if we need more
        if len(all_articles) < min_articles:
            print("\n=== SCRAPING TRIBUN (Backup Source) ===")
            remaining = min_articles - len(all_articles)
            for category in self.categories['tribun'].keys():
                print(f"\nScraping category: {category}")
                articles = self.scrape_tribun(category, remaining // 5 + 3)
                all_articles.extend(articles)
                print(f"Collected {len(articles)} articles from Tribun - {category}")
                if len(all_articles) >= min_articles:
                    break
            
            # Try Kompas if still need more
            if len(all_articles) < min_articles:
                print("\n=== SCRAPING KOMPAS (Additional Source) ===")
                remaining = min_articles - len(all_articles)
                for category in self.categories['kompas'].keys():
                    print(f"\nScraping category: {category}")
                    articles = self.scrape_kompas(category, remaining // 5 + 2)
                    all_articles.extend(articles)
                    print(f"Collected {len(articles)} articles from Kompas - {category}")
                    if len(all_articles) >= min_articles:
                        break
        
        print(f"\n📈 Total articles collected: {len(all_articles)}")
        
        if len(all_articles) >= min_articles:
            print(f"✅ Target reached! Collected {len(all_articles)} out of {min_articles}")
            self.save_to_csv(all_articles)
            print(f"✅ Successfully saved {len(all_articles)} articles to news_articles.csv")
        else:
            print(f"⚠️  Below target. Collected {len(all_articles)} out of {min_articles}")
            if all_articles:
                self.save_to_csv(all_articles)
                print(f"✅ Successfully saved {len(all_articles)} articles to news_articles.csv")
            else:
                print("❌ No articles to save!")
        
        # Print summary
        self.print_summary(all_articles)
        
        print(f"\n🏁 Focused scraping completed! Total articles: {len(all_articles)}")
        return all_articles

if __name__ == "__main__":
    scraper = IndonesianNewsScraper()
    total_articles = scraper.run_async_multithread(150)  # Increased target
    print(f"\n🎉 Asynchronous multi-threaded scraping completed successfully!")
    print(f"📊 Total articles collected: {len(total_articles)}")