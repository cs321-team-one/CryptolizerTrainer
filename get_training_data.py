import pandas as pd
import os
import requests
import json
import datetime

REQUEST_TIMEOUT = 8
DATA_DIRECTORY = './cached_data'
NEWS_DATA_PATH = f'{DATA_DIRECTORY}/news.json'
PRICE_DATA_PATH = f'{DATA_DIRECTORY}/price.json'

OVERWRITE_DATA = False


class NewsAPI:
    search_query = 'cryptocurrency|bitcoin|ethereum|litecoin|neo'
    search_technology = 'technology'
    api_key = '6afbd0e8-5b22-4c72-a97c-ab2040014022'
    base_guardian_url = 'https://content.guardianapis.com'

    def __init__(self,
                 search_query=search_query,
                 search_technology=search_technology,
                 api_key=api_key):
        self.search_query = search_query
        self.search_technology = search_technology
        self.api_key = api_key

    def get_url(self, page_num):
        """
        Constructs the URL for the NewsAPI given a page number.
        :param page_num: the page number for the API.
        :return: The constructed URL for the API.
        """
        return f'{self.base_guardian_url}/search?q={self.search_query}&api-key=' \
               f'{self.api_key}&order-by=relevance&section={self.search_technology}&show-blocks=all&page={page_num}'


class PriceAPI:
    ticker = 'BTC'

    def __init__(self, ticker=ticker):
        self.ticker = ticker

    def get_url(self):
        """
        Constructs the URL for the price API.
        :return: The constructed URL for the API.
        """
        return f'https://min-api.cryptocompare.com/data/histoday?fsym=USD&tsym={self.ticker}&allData=true'


def request_price_data(price_url_or_path, read_from_path):
    """
    Obtains the price data either by requesting it from a URL or reading it from a path.
    :param price_url_or_path: If read_from_path is set to false, price the URL to request the price data.
    :param read_from_path: If set to true, this function will obtain price data from a file path.
    :return: The DataFrame object containing price information.
    """
    price_dataframe = None

    if not read_from_path:
        try:
            price_api_response = requests.get(price_url_or_path, timeout=REQUEST_TIMEOUT).text
            price_api_response = json.loads(price_api_response)
            price_dataframe = pd.DataFrame(price_api_response['Data'])
            price_dataframe = price_dataframe.reindex(
                columns=['time', 'high', 'low', 'open', 'volumefrom', 'volumeto', 'close'])

        except requests.RequestException:
            print('Cannot request the price data API.')
            exit(1)
    else:
        price_dataframe = pd.read_json(price_url_or_path)

    return price_dataframe


def get_training_data():
    """
    Obtains the training data from the AI
    :return: The training data.
    """
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

    price_url = PriceAPI().get_url()

    if not os.path.exists(PRICE_DATA_PATH) or OVERWRITE_DATA:
        price_data = request_price_data(price_url, False)
        with open(PRICE_DATA_PATH, 'w') as f:
            f.write(price_data.to_json(orient='records'))
    else:
        price_data = request_price_data(PRICE_DATA_PATH, True)

    price_data['time'] = price_data['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))

    print(price_data)


def main():
    get_training_data()


if __name__ == "__main__":
    main()
