from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt
def read_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    return req_file.read_text().splitlines() if req_file.exists() else []

setup(
    name="RASCAL",
    version="0.1",
    packages=find_packages(where="src"),  # Ensures it finds src/rascal
    package_dir={"": "src"},  # Specifies src as the package root
    install_requires=read_requirements(),
    include_package_data=True,
)
