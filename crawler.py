import requests
from bs4 import BeautifulSoup
import re
from time import sleep
import random
import os


class crawl_YELP():

    def __init__(self, directory, file_for_anchors):
        self.directory = directory   #путь к папке для скачивания ЧЕРЕЗ \\
        self.root = 'https://www.yelp.com/search?cflt=restaurants&find_loc=San%20Francisco%2C%20CA'
        self.for_anchors = open(file_for_anchors, "a")  # файл для сохранения генерируемых ссылок

    def make_distinct_dirs(self): #выполняется только при первом запуске программы
        os.makedirs(self.directory + '\\positive')
        os.makedirs(self.directory + '\\negative')

    def get_anchor_parts_to_crawl(self, url_address):
        file = requests.get(url_address)
        page = file.text
        soup = BeautifulSoup(page, features="html.parser")
        divs = soup.find_all('div', {'class': 'lemon--div__373c0__1mboc mainAttributes__373c0__1r0QA arrange-unit__373c0__o3tjT arrange-unit-fill__373c0__3Sfw1 border-color--default__373c0__3-ifU'})
        anc = []
        rates = []
        for div in divs:
            one_iter = ['https://www.yelp.com' + c['href'] for c in div.find_all('a') if
                c.has_attr('href') and re.match('^/biz/.+-san-francisco-?\d?$', c['href'])]
            anc += one_iter
            if len(one_iter) != 0:
                rates += [float((el['aria-label'].split(" "))[0]) for el in div.find_all('div') if
                          el.has_attr('aria-label') and re.match('\d\.?\d? star rating', el['aria-label'])]
        for i in range(len(anc)):
            if rates[i] <= 3.5:
                self.for_anchors.write(anc[i] + '\n')

    def crawl_one_address(self, address):
        file = requests.get(address)
        sleep(random.randint(3, 10))
        whole_page = file.text
        soup = BeautifulSoup(whole_page, features="html.parser")
        sleep(random.randint(5, 10))
        divs_stars = soup.find_all('div', {'class': 'lemon--div__373c0__1mboc u-space-t1 u-space-b1 border-color--default__373c0__2oFDT'})
        star_ratings = []
        for el in divs_stars:
            el = str(el)
            sleep(random.randint(3, 5))
            star_ratings.append((re.findall('\d\.?\d? star rating', el)[0].split(" "))[0])
        sleep(random.randint(3, 10))
        spans_texts = soup.findAll('span', lang='en')
        texts = []
        for item in spans_texts:
            item = str(item)
            sleep(10)
            item = re.sub('<[^>]+>', '', item)
            item = item.encode('ascii', 'ignore')
            texts.append(item)
        for i in range(len(texts)):
            if int(star_ratings[i]) > 3:
               sleep(5)
            filename = 'pos_' + str(len(os.listdir(self.directory + '\\positive'))) + '.txt'
            file = open(self.directory + '\\positive' + filename, 'w')
            file.write(texts[i].decode())
            file.close()
        if int(star_ratings[i]) < 3:
            sleep(5)
            filename = 'neg_' + str(len(os.listdir(self.directory + '\\negative'))) + '.txt'
            file = open(self.directory + '\\negative' + filename, 'w')
            file.write(texts[i].decode())
            file.close()
    sleep(random.randint(10, 15))
