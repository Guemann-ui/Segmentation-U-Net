import os
import urllib.request
from zipfile import ZipFile
# download and unzip the dataset
dataset_path = 'dataset1'
# Check if 'dataset1' directory exists, if not, download and unzip the dataset
if not os.path.exists('dataset1'):
    # Download the dataset
    url = 'https://www.dropbox.com/s/0pigmmmynbf9xwq/dataset1.zip'
    urllib.request.urlretrieve(url, 'dataset1.zip')
