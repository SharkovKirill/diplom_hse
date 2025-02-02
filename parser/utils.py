from enum import Enum
import urllib.request
import csv
from datetime import datetime
import os
import uuid

def enum_to_int(v: Enum):
    return v.value

def save_to_directory(fileName: str):
    now = str(datetime.now())
    now_formatted = now.replace(':', '_')
    # getting save dir name
    save_dir = f'images/{now_formatted}'
    # make save dir
    os.makedirs(save_dir)
    
    with open(fileName, 'r') as csvfile:
        filedatareader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in filedatareader:
            url = row[1]
            if(len(url) > 0):
                urllib.request.urlretrieve(url, f'{save_dir}/{uuid.uuid1()}.jpg')