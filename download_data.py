import urllib.request
import re
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def download_dir(base_url, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    req = urllib.request.Request(base_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
    except Exception as e:
        print(f"Failed to read {base_url}: {e}")
        return
    
    # find links
    links = re.findall(r'href="([^"]+)"', html)
    for link in links:
        if link.startswith('?') or link.startswith('/'):
            continue
        if link.endswith('/'):
            download_dir(base_url + link, os.path.join(out_dir, link[:-1]))
        elif link.endswith('.csv.gz'):
            file_url = base_url + link
            file_path = os.path.join(out_dir, link)
            print(f"Downloading {file_url} to {file_path}")
            try:
                urq = urllib.request.Request(file_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(urq) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Failed to download {file_url}: {e}")

download_dir('https://physionet.org/files/mimiciv-demo/2.2/', 'data/mimiciv-demo')
download_dir('https://physionet.org/files/mimic-iv-note-demo/2.2/', 'data/mimic-iv-note-demo')
