import requests
import logging
import os, sys

from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

session = requests.session()
info = open('datasets-info.csv', 'w')
info.write('"name","instances","features","classes"\n')
cnt = 0;
for url in open('datasets-links.txt', 'r'):
    req = session.get(url.strip())
    doc = BeautifulSoup(req.content, 'html.parser')

    dataset_name = doc.find('h1').get_text().strip().replace('.','-').replace(',','-')

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
    logging.info('Collecting info about dataset \'' + dataset_name + '\'')
    n_instances = None
    n_features = None
    n_classes = None
    for panel in doc.find_all(attrs={'class': 'searchresult panel'}):
        name = panel.find('a', {'href': 'a/data-qualities/NumberOfInstances'})
        if name is not None:
            n_instances = panel.find(attrs={'class': 'dataproperty'}).get_text().strip()
        name = panel.find('a', {'href': 'a/data-qualities/NumberOfFeatures'})
        if name is not None:
            n_features = panel.find(attrs={'class': 'dataproperty'}).get_text().strip()
        name = panel.find('a', {'href': 'a/data-qualities/NumberOfClasses'})
        if name is not None:
            n_classes = panel.find(attrs={'class': 'dataproperty'}).get_text().strip()

        if (n_instances is not None and n_features is not None and n_classes is not None):
            break

    info.write(dataset_name + ',' + n_instances + ',' + n_features + ',' + n_classes + '\n')

    logging.info('Complete!')

logging.info('All datasets downloaded successfully!')
