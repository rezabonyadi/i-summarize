import bs4 as bs
import urllib.request
import re
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
    
    

def get_from_page(address='https://en.wikipedia.org/wiki/Artificial_intelligence'):

    scraped_data = urllib.request.urlopen(address)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article,'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text
    
    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)

    ps = article_text.split('\n')
    paragraphs = []
    for k in ps:
        if len(k) < 5:
            continue
        paragraphs.append(re.sub(r'\s+', ' ', k))

    return paragraphs

def get_from_txt(address='data/text_data/singularity.txt'):
    fs = open(address, 'rt', encoding="utf8")
    article_text = fs.read()

    # Removing Square Brackets and Extra Spaces
    # article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)

    ps = article_text.split('\n')
    paragraphs = []
    for k in ps:
        if len(k) < 5:
            continue
        paragraphs.append(k)
        # paragraphs.append(re.sub(r'\s+', ' ', k))

    return paragraphs
