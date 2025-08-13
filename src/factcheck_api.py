import requests

class GoogleFactChecker:
    def __init__(self, api_key):
        self.base_url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        self.api_key = api_key

    def search_claim(self, query):
        params = {
            "query": query,
            "languageCode": "en-US",
            "key": self.api_key
        }
        r = requests.get(self.base_url, params=params)
        if r.status_code == 200:
            return r.json()
        else:
            print("Error:", r.status_code, r.text)
            return None
