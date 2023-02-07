"""
Download datasets for testing
"""
# pylint: disable=C0116
import os
import subprocess

# Paths
test_path = os.path.dirname(__file__)
data_path = os.path.join(test_path, "data")

# Download data
tar_path = os.path.join(test_path, "cf-tools-data.tar.gz")
URL = "https://ndownloader.figshare.com/files/27249593"


def pytest_configure():
    # Download data
    if not os.path.exists(data_path):
        # Download data
        print(f"Downloading data to [{data_path}]", end=" - ")
        commands = [
            f"wget -v -O {tar_path} -L {URL}",
            f"tar xvzf {tar_path} -C {test_path}",
            f"rm -f {tar_path}",
        ]
        subprocess.call(" && ".join(commands), shell=True)
        print("done")
