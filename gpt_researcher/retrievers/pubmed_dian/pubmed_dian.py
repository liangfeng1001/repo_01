# PubmedDian
"""
 @Author: cks
 @Date: 2025/3/8 23:16
 @Description:

"""

# libraries
import os
import requests
import urllib.parse


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
                        }
                        search_response.append(search_result)
                        results_processed += 1
        except Exception as e:
            print(f"Error: {e}. Failed fetching sources. Resulting in empty response.")
            search_response = []

        return search_response
