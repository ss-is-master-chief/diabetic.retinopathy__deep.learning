import zipfile
import os
import requests
from colorama import Fore, Style
from tqdm import tqdm

def download(url, filename):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    file_size = int(requests.head(url).headers["Content-Length"])
    if os.path.exists(filename):
        first_byte = os.path.getsize(filename)
    else:
        first_byte = 0
    if first_byte >= file_size:
        return file_size
    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}
    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])
    req = requests.get(url, headers=header, stream=True)
    with(open(filename, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()

def fetch_db0():    
    if not os.path.exists('diaretdb0_v_1_1.zip'):
        print("{}[+] Downloading diaretdb0_v_1_1.zip..{}".format(Fore.GREEN, Style.RESET_ALL))
        download("http://www.it.lut.fi/project/imageret/diaretdb0/diaretdb0_v_1_1.zip", "diaretdb0_v_1_1.zip")
        print("{}[~] Downloading complete. Extracting..{}".format(Fore.YELLOW, Style.RESET_ALL))
        zip_ref = zipfile.ZipFile('diaretdb0_v_1_1.zip', 'r')
        zip_ref.extractall('.')
        zip_ref.close()
        print("{}[\u2713] Extraction Complete..{}".format(Fore.GREEN, Style.RESET_ALL))
            
def fetch_db1():
    if not os.path.exists('diaretdb1_v_1_1.zip'):
        print("{}[+] Downloading diaretdb1_v_1_1.zip..{}".format(Fore.GREEN, Style.RESET_ALL))
        download("http://www.it.lut.fi/project/imageret/diaretdb1/diaretdb1_v_1_1.zip", "diaretdb1_v_1_1.zip")
        print("{}[+] Downloading complete. Extracting..{}".format(Fore.YELLOW, Style.RESET_ALL))
        zip_ref = zipfile.ZipFile('diaretdb1_v_1_1.zip', 'r')
        zip_ref.extractall('.')
        zip_ref.close()
        print("{}[\u2713] Extraction Complete..{}".format(Fore.GREEN, Style.RESET_ALL))