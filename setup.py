from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Read the long description from README.md
README = (HERE / "README.md").read_text(encoding="utf-8")

# Read requirements.txt
REQS = (HERE / "requirements.txt").read_text().splitlines()

setup(
    name="mlcore",
    version="0.1.0",
    description="Custom implementations of linear and logistic regression",
    long_description=README,
    long_description_content_type="text/markdown",
    author="sushant patil",
    author_email="sushant.kb.patil@gmail.com",
    url="https://github.com/sush4nt/ml-core",  # update as needed
    packages=find_packages(),          # finds the mlcore package
    install_requires=REQS,             # pulls in numpy, etc.
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
