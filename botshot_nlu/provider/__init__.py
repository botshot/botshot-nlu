import requests

from botshot_nlu.dataset.keywords import KeywordDataset


class RestProvider(KeywordDataset):
    
    def __init__(self, api_url):
        if not api_url:
            raise Exception("API URL was not set!")
        self.api_url = api_url
    
    def _get(self):
        headers = {"Content-Type": "application/json"}
        resp = requests.get(self.api_url, headers=headers)
        data = resp.json()
        if not isinstance(data, dict):
            print("Invalid response, expected dict but got {}".format(type(data)))
            return {}
        return data

    def get_entities(self):
        data = self._get()
        return set(data.keys())

    def get_data(self, entities):
        data = self._get()
        return {k: data[k] for k in data.keys() & entities}

    def has_changed(self, entities):
        return True  # TODO
