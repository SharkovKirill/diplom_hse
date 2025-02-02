from dotenv import load_dotenv
import pandas as pd
from selenium import webdriver 
from selenium.webdriver.common.by import By
import os
import urllib.parse
from urllib.parse import urljoin, urlencode
import time

from districts import District
from object_type import ObjectType
from utils import enum_to_int

DEFAULT_PATH="kupit-kvartiru-1-komn-ili-2-komn"
SLEEP_TIME=5

class SimpleCianParser():
    # CIAN url
    __url = None
    
    def __init__(self):
        # take environment variables from .env
        load_dotenv()
        # take CIAN_URL variavle from .env
        self.__url = os.environ.get('CIAN_URL', "https://www.cian.ru")
        
    def build_url(
            self, 
            path: str = DEFAULT_PATH, 
            params: dict = {}
        ) -> str:
        query = "?" + urlencode(params)
        
        return urljoin(self.__url, path + query)
    
    def parse(self, url: str, images_cnt: int, pages_cnt: int) -> pd.DataFrame:
        # create the pandas DataFrame
        df = pd.DataFrame([], columns=['url'])
        # create the driver
        driver = webdriver.Chrome()
        # loading the web page
        driver.get(url)
        
        for page in range(1, pages_cnt):
            # finding flat list component
            # flat_list_ex = "/html/body/div[1]/div/div[6]"
            # flat_list = driver.find_element(By.XPATH, flat_list_ex)
            
            # getting flat component
            for flat in driver.find_elements(By.TAG_NAME, 'article'):

                # getting images list component
                # img_list = flat.find_element(By.XPATH, '//article/div[1]/a/div[1]/div/ul')
                try: 
                    ul = flat.find_element(By.TAG_NAME, 'ul')
                    
                    for img in ul.find_elements(By.TAG_NAME, 'img'):
                        src = img.get_attribute('src')
                        
                        df.loc[-1] = src
                        df.index = df.index + 1
                        df = df.sort_index()
                        
                        if df.shape[0] == images_cnt:
                            return df
                except:
                    print('element doesnt exist')
                    
            if(page == pages_cnt):
                return df
            
            # finding next page button 
            next_button = driver.find_element(By.XPATH, "//*[contains(text(), 'Дальше')]")
            # clicking next page button
            next_button.click()
            # sleep in order to avoid blocking
            time.sleep(SLEEP_TIME)
        # Closes the browser 
        driver.quit()
        
        return df
    
    def get_flat_images(
        self, 
        images_cnt = 100,
        pages_cnt = 10,
        districts: list[District] = [],
        object_types: list[ObjectType] = [],
        maxprice = None,
        minprice = None
    ) -> pd.DataFrame:
        url = ""
        
        if(len(districts) == 0):
            url = self.build_url()
        else:
            # setting params
            params = dict(
                deal_type = 'sale',
                engine_version = 2,
                offer_type = 'flat'
            )
            # todo: remove this loop
            for idx, d in enumerate(districts):
                params[f'district[{idx}]'] = enum_to_int(d)
                
            for idx, o in enumerate(object_types):
                params[f'object_type[{idx}]'] = enum_to_int(o)
                
            if(maxprice != None):
                params['maxprice'] = maxprice
            
            if(minprice != None):
                params['minprice'] = minprice
                
            # build the url with params
            url = self.build_url(path="cat.php", params=params)
        
        # parsing thr website
        result = self.parse(url, images_cnt, pages_cnt)
        
        return result
       
        