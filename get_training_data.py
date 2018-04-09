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
        return f'{self.base_guardian_url}/search?' \
               f'q={self.search_query}' \
               f'&api-key={self.api_key}' \
               f'&order-by=relevance' \
               f'&section={self.search_technology}' \
               f'&show-blocks=all' \
               f'&page={page_num}' \
               f'&page-size=50' \
               f'&from-date=2013-04-03'


class PriceAPI:
    ticker = 'BTC'

    def __init__(self, ticker=ticker):
        self.ticker = ticker

    def get_url(self):
        """
        Constructs the URL for the price API.
        :return: The constructed URL for the API.
        """
        return f'https://min-api.cryptocompare.com/data/histoday?' \
               f'fsym=USD' \
               f'&tsym={self.ticker}' \
               f'&allData=true'


def process_price_api_time(x):
    now = datetime.datetime.fromtimestamp(x)
    return datetime_keep_only_date(now)


def request_price_data_from_api(price_api):
    """
    Takes the URL from the given price api object and requests the price data.
    :param price_api:
    :return:
    """
    price_url = price_api.get_url()
    price_api_response = requests.get(price_url, timeout=REQUEST_TIMEOUT).text
    price_data = json.loads(price_api_response)['Data']
    print(f'Successfully requested data from the price api with {len(price_data)} entries.')
    return price_data


def read_price_data_from_path(price_data_path):
    """
    Obtains the price data from a file path.
    :param price_data_path: The path to the JSON file containing price data.
    :return: The DataFrame object containing price information.
    """
    with open(price_data_path, 'r') as json_data:
        data = json.load(json_data)

    return data


def process_news_api_time(x):
    now = datetime.datetime.strptime(x.split('T')[0], '%Y-%m-%d')
    return now


def request_news_data_from_api(news_api):
    # Perform the initial request to get the number of pages to request

    news_url = news_api.get_url(1)
    initial_news_api_response = requests.get(news_url, timeout=REQUEST_TIMEOUT).text
    initial_news_api_response = json.loads(initial_news_api_response)

    news_data = []

    num_pages = initial_news_api_response['response']['pages']

    for page in range(1, num_pages):
        news_api_response = requests.get(news_api.get_url(page)).text
        news_api_response = json.loads(news_api_response)['response']['results']
        for article in news_api_response:
            if len(article['blocks']['body']) > 0:
                news_data.append({
                    'time': article['webPublicationDate'],
                    'title': article['webTitle'],
                    'text': article['blocks']['body'][0]['bodyTextSummary']
                })
        print(f'Successfully requested {page}/{num_pages} article blocks.')

    return news_data


def read_news_data_from_path(news_data_path):
    with open(news_data_path, 'r') as json_data:
        data = json.load(json_data)

    return data


def datetime_keep_only_date(datetime_with_time):
    return datetime.datetime(datetime_with_time.year, datetime_with_time.month, datetime_with_time.day)


def combine_price_and_news_data(price_data, news_data):

    # Loop through the news articles and append the corresponding change in the day's price
    found_prices = 0

    price_change_array = []
    for current_day, text, title in zip(news_data['time'], news_data['text'], news_data['title']):
        for price_time, price_open, price_close in zip(price_data['time'], price_data['open'], price_data['close']):
            if price_time == current_day:
                price_change = (price_close - price_open) / price_open
                price_change_array.append(price_change)
                found_prices += 1

    news_data['price_change'] = pd.Series(price_change_array)

    return news_data


def process_price_data(price_data):
    """
    Takes in a JSON dict, converts it to a Pandas DataFrame, and pre-processes the data.
    :param price_data: JSON dict
    :return: Processed DataFrame
    """
    price_data = pd.DataFrame(price_data)
    price_data = pd.DataFrame(price_data)
    price_data = price_data.reindex(
        columns=['time', 'high', 'low', 'open', 'volumefrom', 'volumeto', 'close'])
    price_data['time'] = price_data['time'].apply(lambda x: process_price_api_time(x))
    return price_data


def process_news_data(news_data):
    combined_news_data = []
    while len(news_data) > 0:
        current_article = news_data.pop()

        current_article_date = current_article['time'].split('T')[0]
        combined_article = {
            'time': current_article['time'],
            'title': current_article['title'],
            'text': current_article['text']
        }
        for compare_news in news_data:
            if compare_news['time'].split('T')[0] == current_article_date:
                combined_article['title'] += ' ' + compare_news['title']
                combined_article['text'] += ' ' + compare_news['text']
                news_data.remove(compare_news)
        combined_news_data.append(combined_article)

    combined_news_data = pd.DataFrame(combined_news_data)
    combined_news_data['time'] = combined_news_data['time'].apply(lambda x: process_news_api_time(x))

    return combined_news_data


def get_training_data():
    """
    Obtains the training data from the AI
    :return: The training data.
    """
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

    price_api = PriceAPI()
    news_api = NewsAPI()

    price_data = None
    news_data = None

    # Get the price data
    if not os.path.exists(PRICE_DATA_PATH) or OVERWRITE_DATA:
        try:
            price_data = request_price_data_from_api(price_api)
            with open(PRICE_DATA_PATH, 'w') as f:
                json.dump(price_data, f)
        except requests.RequestException:
            print('Error obtaining data from the API... Attempting to read cached data...')
            if os.path.exists(PRICE_DATA_PATH):
                price_data = read_price_data_from_path(PRICE_DATA_PATH)
    elif os.path.exists(PRICE_DATA_PATH):
        price_data = read_price_data_from_path(PRICE_DATA_PATH)

    # Get the news data
    if not os.path.exists(NEWS_DATA_PATH) or OVERWRITE_DATA:
        try:
            news_data = request_news_data_from_api(news_api)
            with open(NEWS_DATA_PATH, 'w') as f:
                json.dump(news_data, f)
        except requests.RequestException:
            print('Error obtaining data from the news API... Attempting to read cached data...')
            if os.path.exists(NEWS_DATA_PATH):
                news_data = read_news_data_from_path(NEWS_DATA_PATH)
    elif os.path.exists(NEWS_DATA_PATH):
        news_data = read_news_data_from_path(NEWS_DATA_PATH)

    price_data = process_price_data(price_data)
    news_data = process_news_data(news_data)

    price_news_data = combine_price_and_news_data(price_data, news_data)

    return price_news_data


def main():
    training_data = get_training_data()
    print(training_data)


if __name__ == "__main__":
    main()
