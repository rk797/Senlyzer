import requests as r
from bs4 import BeautifulSoup
import pandas as pd

from tqdm.auto import tqdm

url_init= "https://www.noslang.com/dictionary/"

url_add_ons=["1/","a/","b/","c/", "d/", "e/", "f/", "g/", "h/", "i/", "j/", "k/", "l/", "m/", "n/", "o/", "p/", "q/", "r/","s/", "t/", "u/", "v/", "w/", "x/", "y/", "z/"]

url_list=[]

for i in url_add_ons:
    url_list.append(url_init+i)

terms=[]
definitions=[]

for url in tqdm(url_list):
    response = r.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    abbr_words= soup.find_all('dt')
    for word in abbr_words:
        term = word.text[0:-2]
        terms.append(term)
    
    
    defs = soup.find_all('dd')
    for i in defs:
        definitions.append(i.text)

print(terms[:5])
print(definitions[:5])

my_dict= dict(zip(terms, definitions))
print(my_dict)

df= pd.DataFrame(my_dict.items(), columns=["Slang","Meaning"])

df.to_csv("slangs_processing/internet-slangs-to-normal-terms.csv")
    
