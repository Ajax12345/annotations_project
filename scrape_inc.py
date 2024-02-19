import requests, json, time
import re, urllib.parse

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


def crawl_companies() -> None:
    stream = company_stream()
    for _ in range(1000):
        print(next(stream))

if __name__ == '__main__':
    crawl_companies()