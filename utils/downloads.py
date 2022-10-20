"""
Download utils
"""

import os
import platform
from random import shuffle
import shutil
import subprocess
import tempfile
import time
import urllib
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import oneflow as flow
import requests
from tqdm import tqdm


def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=""):
    # gs://bucket/file size https://cloud.google.com/storage/docs/gsutil/commands/du
    s = subprocess.check_output(f"gsutil du {url}", shell=True).decode("utf-8")
    return eval(s.split(" ")[0]) if len(s) else 0  # bytes

def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f"Downloading {url} to {file}...")
        flow.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print("")


def attempt_download(file, repo="Oneflow-Inc/one-yolov5"):  # from utils.downloads import *; attempt_download()
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", ""))

    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                print(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        # GitHub assets
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
        try:
            response = requests.get(f"https://api.github.com/repos/{repo}/releases/latest").json()  # github api
            assets = [x["name"] for x in response["assets"]]  # release assets, i.e. ['yolov5s', 'yolov5m', ...]
            tag = response["tag_name"]  # i.e. 'v1.0'
        except:  # fallback plan
            assets = [
                "yolov5n.zip",
                "yolov5s.zip",
                "yolov5m.zip",
                "yolov5l.zip",
                "yolov5x.zip",
                "yolov5n6.zip",
                "yolov5s6.zip",
                "yolov5m6.zip",
                "yolov5l6.zip",
                "yolov5x6.zip",
            ]
            try:
                tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except:
                tag = "v1.0"  # current release

        name = name + ".zip"
        file = Path(name)
        if name in assets:
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # backup url (optional)
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/",
            )

        new_dir = Path(name[:-4])
        if not os.path.exists(new_dir): # 判断文件夹是否存在
            os.mkdir(new_dir)# 新建文件夹

        if ".zip" in name:
            print("unzipping... ", end="")
            # ZipFile(new_file).extractall(path=file.parent)  # unzip
            f = ZipFile(file)
            f.extractall(new_dir)
            os.remove(file)  # remove zip
            tmp_dir = "/tmp/oneyolov5"
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)

            path1 = os.path.join(name[:-4], name[:-4])
            shutil.copytree(path1, tmp_dir)
            shutil.rmtree(new_dir)
            shutil.copytree(tmp_dir, new_dir)
            shutil.rmtree(tmp_dir)

    return str(file)


def gdrive_download(id="16TiPfZj7htmTyhntwcZyEEAejOUxuT6m", file="tmp.zip"):
    # Downloads a file from Google Drive. from yolov5.utils.downloads import *; gdrive_download()
    t = time.time()
    file = Path(file)
    cookie = Path("cookie")  # gdrive cookie
    print(f"Downloading https://drive.google.com/uc?export=download&id={id} as {file}... ", end="")
    file.unlink(missing_ok=True)  # remove existing file
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Attempt file download
    out = "NUL" if platform.system() == "Windows" else "/dev/null"
    os.system(f'curl -c ./cookie -s -L "drive.google.com/uc?export=download&id={id}" > {out}')
    if os.path.exists("cookie"):  # large file
        s = f'curl -Lb ./cookie "drive.google.com/uc?export=download&confirm={get_token()}&id={id}" -o {file}'
    else:  # small file
        s = f'curl -s -L -o {file} "drive.google.com/uc?export=download&id={id}"'
    r = os.system(s)  # execute, capture return
    cookie.unlink(missing_ok=True)  # remove existing cookie

    # Error check
    if r != 0:
        file.unlink(missing_ok=True)  # remove partial
        print("Download error ")  # raise Exception('Download error')
        return r

    # Unzip if archive
    if file.suffix == ".zip":
        print("unzipping... ", end="")
        ZipFile(file).extractall(path=file.parent)  # unzip
        file.unlink()  # remove zip

    print(f"Done ({time.time() - t:.1f}s)")
    return r


def get_token(cookie="./cookie"):
    with open(cookie) as f:
        for line in f:
            if "download" in line:
                return line.split()[-1]
    return ""


# Google utils: https://cloud.google.com/storage/docs/reference/libraries ----------------------------------------------
#
#
# def upload_blob(bucket_name, source_file_name, destination_blob_name):
#     # Uploads a file to a bucket
#     # https://cloud.google.com/storage/docs/uploading-objects#storage-upload-object-python
#
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(destination_blob_name)
#
#     blob.upload_from_filename(source_file_name)
#
#     print('File {} uploaded to {}.'.format(
#         source_file_name,
#         destination_blob_name))
#
#
# def download_blob(bucket_name, source_blob_name, destination_file_name):
#     # Uploads a blob from a bucket
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#
#     blob.download_to_filename(destination_file_name)
#
#     print('Blob {} downloaded to {}.'.format(
#         source_blob_name,
#         destination_file_name))
