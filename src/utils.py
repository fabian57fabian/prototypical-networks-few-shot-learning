import os


def download_file_from_url(url: str, dest_dir: str):
    """
    @param url: Web url to download
    @param dest_dir: Destination directory for file
    """
    os.system(f'wget -q "{url}" -P {dest_dir}')