from biorxiv_retriever import BiorxivRetriever
import urllib.request
import time
import random
from urllib.error import HTTPError, URLError

class bioRxivSearch:
    """
    bioRxiv API Retriever
    """
    def __init__(self, query, sort='Relevance', query_domains=None):
        self.query = query
        assert sort in ['Relevance', 'SubmittedDate'], "Invalid sort criterion"
        self.sort = 'date' if sort == 'SubmittedDate' else 'relevance'
        
        # 初始化请求配置
        self._set_custom_headers()
        self.client = BiorxivRetriever(search_engine='biorxiv')
        
        # 重试参数
        self.max_retries = 3
        self.retry_delay = 5  # 基础延迟时间
        self.timeout = 10  # 请求超时时间

    def _set_custom_headers(self):
        """设置自定义请求头"""
        opener = urllib.request.build_opener()
        opener.addheaders = [
            # ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'),
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0'),
            ('Accept-Language', 'en-US,en;q=0.9'),
            ('Referer', 'https://www.biorxiv.org/')
        ]
        urllib.request.install_opener(opener)

    def _query_with_retry(self):
        """带重试机制的查询"""
        for attempt in range(self.max_retries):
            try:
                return self.client.query(
                    self.query,
                    metadata=True,
                    full_text=False
                )
            except (HTTPError, URLError) as e:
                if hasattr(e, 'code') and e.code in [403, 504]:
                    delay = self.retry_delay * (attempt + 1) + random.uniform(0, 2)
                    print(f"Attempt {attempt+1} failed. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                raise
        raise Exception(f"Failed after {self.max_retries} retries")

    def search(self, max_results=5):
        """
        执行搜索
        :param max_results: 最大返回结果数
        :return: 搜索结果列表
        """
        try:
            papers = self._query_with_retry()
            
            search_results = []
            count = 0
            
            for paper in papers:
                if count >= max_results:
                    break
                
                search_results.append({
                    "title": paper.get('title', 'No Title'),
                    "href": paper.get('biorxiv_url', '#'),
                    "body": paper.get('abstract', 'No Abstract Available'),
                })
                count += 1
            
            return search_results
        
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []