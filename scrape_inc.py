import requests, json, time, bs4
import re, urllib.parse, typing
from bs4 import BeautifulSoup as soup
import collections, asyncio, aiohttp
import csv

headers = {
    'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.inc.com/',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'sec-ch-ua-platform': '"macOS"',
}

params = {
    'records': '1000',
    'page': '1',
}

def extract_companies() -> None:
    for i in range(10):
        response = requests.get('https://api.inc.com/rest/i5list/2023', params={**params, 'page': str(i)}, headers=headers).json()
        with open('company_data.json') as f:
            data = json.load(f)
            data.append(response)
        
        with open('company_data.json', 'w') as f:
            json.dump(data, f)
        
        time.sleep(2)

def company_stream() -> None:
    with open('company_data.json') as f:
        data = json.load(f)
    
    for i in data:
        match i:
            case {'companies':[*companies]} if companies:
                for company in companies:
                    if 'website' in company:
                        yield f'https://{company["website"]}' if not urllib.parse.urlparse(company["website"]).scheme else company["website"]


def traverse_src(src:soup) -> typing.Iterator:
    if isinstance(src, bs4.element.NavigableString):
        if src.strip():
            yield src.strip()
        return 
    for i in getattr(src, 'contents', []):
        if i.name is None or i.name.lower() not in ['header', 'nav', 'footer', 'script', 'style']:
            yield from traverse_src(i)

def scrape_text(src:str, url:str) -> dict:
    d = collections.defaultdict(list)
    for t in ['nav', 'footer']:
        for i in soup(src, 'html.parser').select(f'{t} a'):
            d[t].append(text:=i.get_text(strip=True))
            for j in ['team', 'about', 'about us', 'people', 'leadership']:
                if j in text.lower() and 'href' in i.attrs:
                    d[j].append(urllib.parse.urljoin(url, i['href']))
        
    return d

async def run_request(session, url:str) -> tuple:
    try:
        async with session.get(url, headers = headers) as response:
            return ((await response.text()).encode('ascii', 'ignore').decode('ascii'), url)
    except:
        print('fail', url)
        return (None, None)

async def crawl_companies() -> None:
    stream = company_stream()
    async with aiohttp.ClientSession() as session:
        for _ in range(10):
            pages = await asyncio.gather(*[run_request(session, next(stream)) for _ in range(100)])
            about = []
            for page, url in pages:
                print(url)
                if page is None:
                    continue

                d = scrape_text(page, url)
                with open('nav_footers.csv', 'a') as f:
                    write = csv.writer(f)
                    result = 0
                    for i in ['nav', 'footer']:
                        write.writerows(k:={(url, i, j) for j in d.get(i, []) if j})
                        result = result or k

                    

                for i in ['team', 'people', 'leadership', 'about', 'about us']:
                    if i in d:
                        about.append(d[i][0])
                        break

            print(about)
            about_pages = await asyncio.gather(*[run_request(session, i) for i in about])
            for page, url in about_pages:
                if page is None:
                    continue

                with open('about_data.csv', 'a') as f:
                    write = csv.writer(f)
                    write.writerows([[url, i] for i in traverse_src(soup(page, 'html.parser'))])



if __name__ == '__main__':
    asyncio.run(crawl_companies())