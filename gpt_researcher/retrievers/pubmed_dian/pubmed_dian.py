# PubmedDian
"""
 @Author: cks
 @Date: 2025/3/8 23:16
 @Description:

"""

"""
Date: added by cks at 20250610:
Description: 
    added new fields eg. [authors,published,vol,pagination],

    eg like : 2021 Apr;115:574-590
              --------|---|-------
                  ⬆     ⬆     ⬆
            published  vol  pagination

    authors : Short author name list. eg : [McDaniels JM, Huckaby AC, Francis A].
    published : The date the article was published. eg : [2006 Dec].
    vol : Volume number of the journal. eg : [18].
    pagination : The full pagination of the article. eg : [658-65].

"""

# libraries
import os
import requests
import urllib.parse
import logging

logger = logging.getLogger(__name__)


class PubmedDianSearch():
    """
    PubmedDianSearch
    """

    def __init__(self, query, query_domains=None):
        """
        Initializes the PubmedDianSearch object
        Args:
            query:
        """
        self.query = query
        self.api_domain = self.get_api_domain()
        self.api_key = self.get_api_key()

    def get_api_domain(self):
        """
        Gets the PubmedDianSearch API Domain
        Returns:
            like
            http://172.26.230.81:9400
            https://172.26.230.81
            https://www.x.cn

        """
        try:
            api_domain = os.environ["PUBMEDDIAN_API_DOMAIN"]
        except:
            raise Exception("PubmedDianSearch domain not found. Please set the PUBMEDDIAN_API_DOMAIN environment variable. "
                            "You can get a info at yusm1@dazd.cn")
        return api_domain

    def get_api_key(self):
        """
        Gets the PubmedDianSearch API key
        Returns:

        """
        try:
            api_key = os.environ["PUBMEDDIAN_API_KEY"]
        except:
            raise Exception("PubmedDianSearch key not found. Please set the PUBMEDDIAN_API_KEY environment variable. "
                            "You can get a key at yusm1@dazd.cn")
        return api_key

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        print("PubmedDianSearch: Searching with query {0}...".format(self.query))
        """Useful for general internet search queries using PubmedDian."""

        """

        参数:
            query: 搜索查询词。
            max_results: 要返回的最大结果数。默认值为 7。
        返回值:
            一个搜索结果列表。每个结果是一个包含以下键的字典：
            - title: 搜索结果的标题。
            - href: 搜索结果的 URL。
            - body: 搜索结果的正文。
        """

        url = f"{self.api_domain}/api/v1/getPubMedDianSearch"
        params = {
            "q": self.query,
            "from": "gpt",
        }

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'X-PubmedDianApi-Source': 'gpt-researcher'
        }

        encoded_url = url + "?" + urllib.parse.urlencode(params)
        print(f"Encoded URL: {encoded_url}")
        search_response = []

        try:
            response = requests.get(encoded_url, headers=headers, timeout=80)
            if response.status_code == 200:
                search_results = response.json()
                if search_results:
                    results = search_results["data"]
                    results_processed = 0
                    for result in results:
                        if results_processed >= max_results:
                            break
                        if not "AB" in result or not "TI" in result:
                            continue
                        search_result = {
                            "title": result["TI"],
                            "href": result["LINK"],
                            "body": result["AB"],
                            "authors": result["AU"] if "AU" in result else "",
                            "published": result["DP"] if "DP" in result else "",
                            "vol": result["VI"] if "VI" in result else "",
                            "pagination": result["PG"] if "PG" in result else "",
                        }
                        search_response.append(search_result)
                        results_processed += 1


        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            logger.info(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []

        return search_response
