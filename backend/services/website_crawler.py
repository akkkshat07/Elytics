import asyncio
import logging
import re
from typing import Dict, List, Optional, Set, Any
from urllib.parse import urljoin, urlparse
from datetime import datetime
import httpx
from bs4 import BeautifulSoup
logger = logging.getLogger(__name__)

class WebsiteCrawler:

    def __init__(self, max_pages: int=15, timeout: int=15, max_depth: int=3, delay: float=0.3, max_concurrent: int=5):
        self.max_pages = max_pages
        self.timeout = timeout
        self.max_depth = max_depth
        self.delay = delay
        self.max_concurrent = max_concurrent
        self.visited_urls: Set[str] = set()
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

    def _normalize_url(self, url: str) -> str:
        url = url.strip()
        if not url:
            return url
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
        return url

    def _identify_page_type(self, url: str, title: str, content: str) -> str:
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content[:500].lower()
        if url_lower in ['/', '/index', '/index.html', '/home'] or not url_lower.replace('/', '').replace('www.', '').replace('http://', '').replace('https://', '').strip():
            return 'homepage'
        if any((keyword in url_lower or keyword in title_lower for keyword in ['about', 'company', 'who-we-are', 'our-story', 'history', 'mission', 'vision', 'team'])):
            return 'about'
        if any((keyword in url_lower or keyword in title_lower for keyword in ['product', 'catalog', 'shop', 'store', 'item', 'collection'])):
            return 'products'
        if any((keyword in url_lower or keyword in title_lower for keyword in ['service', 'solution', 'offer', 'capability'])):
            return 'services'
        if any((keyword in url_lower or keyword in title_lower for keyword in ['contact', 'reach', 'location', 'office', 'address'])):
            return 'contact'
        if any((keyword in url_lower or keyword in title_lower for keyword in ['review', 'testimonial', 'feedback', 'rating', 'customer-story'])):
            return 'reviews'
        if any((keyword in url_lower or keyword in title_lower for keyword in ['blog', 'news', 'article', 'editorial', 'press', 'media'])):
            return 'blog'
        return 'other'

    def _extract_clean_text(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, 'lxml')
            for element in soup(['script', 'style']):
                element.decompose()
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.main-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            content_source = main_content if main_content else soup
            text = content_source.get_text(separator=' ', strip=True)
            text = re.sub('\\s+', ' ', text)
            lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
            text = ' '.join(lines)
            return text.strip()
        except Exception as e:
            logger.warning(f'Error extracting text: {e}')
            return ''

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        try:
            soup = BeautifulSoup(html, 'lxml')
            links = []
            base_domain = urlparse(base_url).netloc
            link_sources = []
            for anchor in soup.find_all('a', href=True):
                href = anchor.get('href')
                if href:
                    link_sources.append(href)
            for elem in soup.find_all(['div', 'span', 'button'], attrs={'data-href': True}):
                href = elem.get('data-href')
                if href:
                    link_sources.append(href)
            for href in link_sources:
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue
                absolute_url = urljoin(base_url, href)
                parsed = urlparse(absolute_url)
                if parsed.netloc == base_domain and parsed.scheme in ['http', 'https']:
                    normalized = f'{parsed.scheme}://{parsed.netloc}{parsed.path}'
                    if parsed.query:
                        normalized += f'?{parsed.query}'
                    if normalized not in links:
                        links.append(normalized)
            logger.debug(f'Extracted {len(links)} links from {base_url}')
            return links
        except Exception as e:
            logger.warning(f'Error extracting links: {e}')
            return []

    def _prioritize_urls(self, urls: List[str], base_url: str) -> List[str]:
        priority_keywords = {'homepage': ['/', '/index', '/home'], 'about': ['about', 'company', 'who-we-are', 'our-story', 'history'], 'products': ['product', 'catalog', 'shop', 'collection'], 'services': ['service', 'solution', 'offer'], 'contact': ['contact', 'reach'], 'reviews': ['review', 'testimonial', 'feedback'], 'blog': ['blog', 'news', 'article']}

        def get_priority(url: str) -> int:
            url_lower = url.lower()
            for priority, keywords in enumerate(priority_keywords.values()):
                if any((kw in url_lower for kw in keywords)):
                    return priority
            return 999
        return sorted(urls, key=get_priority)

    async def _crawl_page(self, url: str, client: httpx.AsyncClient, depth: int=0) -> Optional[Dict[str, Any]]:
        if url in self.visited_urls or depth > self.max_depth:
            return None
        try:
            logger.info(f'Crawling: {url} (depth: {depth})')
            self.visited_urls.add(url)
            response = await client.get(url, headers={'User-Agent': self.user_agent})
            final_url = str(response.url)
            if final_url != url:
                logger.info(f'Redirected from {url} to {final_url}')
                self.visited_urls.add(final_url)
            if response.status_code != 200:
                logger.warning(f'Failed to fetch {url}: Status {response.status_code}')
                return None
            html = response.text
            if not html or len(html) < 100:
                logger.warning(f'Received empty or very short content from {url} (length: {(len(html) if html else 0)})')
                return None
            logger.debug(f'Fetched {len(html)} bytes from {url}')
            soup = BeautifulSoup(html, 'lxml')
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ''
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            meta_description = meta_desc.get('content', '') if meta_desc else ''
            text_content = self._extract_clean_text(html)
            if not text_content or len(text_content.strip()) < 50:
                logger.warning(f'No meaningful content extracted from {url}')
                return None
            page_type = self._identify_page_type(url, title_text, text_content)
            links = self._extract_links(html, url) if depth < self.max_depth else []
            return {'url': url, 'title': title_text, 'meta_description': meta_description, 'content': text_content, 'page_type': page_type, 'links': links, 'depth': depth}
        except asyncio.TimeoutError:
            logger.warning(f'Timeout crawling {url}')
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(f'HTTP error crawling {url}: {e.response.status_code}')
            return None
        except httpx.RequestError as e:
            logger.warning(f'Request error crawling {url}: {e}')
            return None
        except Exception as e:
            logger.warning(f'Error crawling {url}: {e}')
            return None

    async def crawl_website(self, website_url: str) -> Dict[str, Any]:
        result = {'success': False, 'url': website_url, 'pages_crawled': 0, 'pages': [], 'error': None}
        try:
            website_url = self._normalize_url(website_url)
            result['url'] = website_url
            parsed = urlparse(website_url)
            if not parsed.scheme or not parsed.netloc:
                result['error'] = 'Invalid URL format'
                return result
            base_url = f'{parsed.scheme}://{parsed.netloc}'
            logger.info(f'Starting website crawl: {website_url}')
            timeout_config = httpx.Timeout(self.timeout, connect=5.0, read=self.timeout)
            async with httpx.AsyncClient(timeout=timeout_config, follow_redirects=True, limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)) as client:
                pages_to_crawl = [(website_url, 0)]
                crawled_pages: List[Dict] = []

                async def crawl_batch(urls_to_crawl: List[tuple]) -> List[Dict]:
                    tasks = [self._crawl_page(url, client, depth) for url, depth in urls_to_crawl]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    valid_pages = []
                    for result in results:
                        if isinstance(result, dict) and result:
                            valid_pages.append(result)
                        elif isinstance(result, Exception):
                            logger.debug(f'Exception during batch crawl: {result}')
                    return valid_pages
                page_types_found = set()
                while pages_to_crawl and len(crawled_pages) < self.max_pages:
                    batch_size = min(self.max_concurrent, len(pages_to_crawl), self.max_pages - len(crawled_pages))
                    batch = [pages_to_crawl.pop(0) for _ in range(batch_size)]
                    batch_results = await crawl_batch(batch)
                    new_links_to_add = []
                    for page_data in batch_results:
                        if page_data:
                            crawled_pages.append(page_data)
                            page_types_found.add(page_data.get('page_type', 'other'))
                            if page_data.get('depth', 0) < self.max_depth:
                                new_links = [link for link in page_data.get('links', []) if link not in self.visited_urls and len(crawled_pages) < self.max_pages]
                                if new_links:
                                    prioritized_links = self._prioritize_urls(new_links, base_url)
                                    links_to_add = prioritized_links[:20]
                                    for link in links_to_add:
                                        if link not in [p[0] for p in pages_to_crawl] and link not in [p[0] for p in new_links_to_add]:
                                            new_links_to_add.append((link, page_data.get('depth', 0) + 1))
                                    logger.debug(f"Added {len(links_to_add)} links to crawl queue from {page_data.get('url')}")
                    pages_to_crawl.extend(new_links_to_add)
                    if len(page_types_found) >= 5 and len(crawled_pages) >= 8:
                        logger.info(f'Found diverse content ({len(page_types_found)} types, {len(crawled_pages)} pages). Stopping early.')
                        break
                    if pages_to_crawl and len(crawled_pages) < self.max_pages:
                        await asyncio.sleep(self.delay)
                if crawled_pages:
                    result['success'] = True
                    result['pages_crawled'] = len(crawled_pages)
                    result['pages'] = crawled_pages
                    logger.info(f'Successfully crawled {len(crawled_pages)} pages from {website_url}')
                else:
                    error_details = []
                    if len(self.visited_urls) == 0:
                        error_details.append('Could not fetch homepage - website may be blocking crawlers or require JavaScript')
                    else:
                        error_details.append(f'Visited {len(self.visited_urls)} URLs but extracted no content - pages may be empty or blocked')
                    result['error'] = 'No pages could be crawled. ' + ' '.join(error_details)
                    logger.warning(f"Failed to crawl any pages from {website_url}: {result['error']}")
        except httpx.TimeoutException as e:
            logger.error(f'Timeout during website crawl: {e}')
            result['error'] = f'Request timeout - website took too long to respond (>{self.timeout}s)'
        except httpx.ConnectError as e:
            logger.error(f'Connection error during website crawl: {e}')
            result['error'] = f'Connection failed - website may be unreachable or blocking requests'
        except Exception as e:
            logger.error(f'Error during website crawl: {e}', exc_info=True)
            result['error'] = f'Unexpected error: {str(e)}'
        return result