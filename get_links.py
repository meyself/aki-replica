import urllib.request
import re

def get_links(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    links = re.findall(r'href=[\'"]([^\'"]*zip)[\'"]', html)
    print("Links for", url, ":", set(links))

try:
    get_links('https://physionet.org/content/mimic-iv-demo/2.2/')
    get_links('https://physionet.org/content/mimic-iv-note-demo/2.2/')
except Exception as e:
    print(e)
