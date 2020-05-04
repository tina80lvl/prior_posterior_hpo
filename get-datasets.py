import requests
import logging
import os, sys

from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

session = requests.session()

for url in open('datasets-links.txt', 'r'):
    req = session.get(url.strip())
    doc = BeautifulSoup(req.content, 'html.parser')

    dataset_name = doc.find('h1').get_text().strip()

    if (dataset_name + '.csv' in os.listdir('./datasets/')):
        logging.info('Skipping dataset \'' + dataset_name + '\': already exists')
        continue

    # Downloading dataset
    logging.info('Downloading dataset \'' + dataset_name + '\'')
    for link in doc.find_all('a', {'onclick': 'doDownload()'}):
        download_url = link.get('href')
        if ('csv' in download_url):
            open('./datasets/' + dataset_name + '.csv', 'wb').write(
                requests.get(download_url).content
            )

    # Collecting information about dataset
    # TODO
    logging.info('Complete!')

logging.info('All datasets downloaded successfully!')
