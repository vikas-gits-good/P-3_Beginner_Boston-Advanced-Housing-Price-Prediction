from setuptools import find_packages, setup
from typing import List

__version__ = "0.1.0"
Repo_name = "P-3_Beginner_Boston-Advanced-Housing-Price-Prediction"
Author_name = "vikas-gits-good"
Author_email = "vikas.c.conappa@protonmail.com"


def get_packages(File_path: str = "requirements.txt") -> List[str]:
    """
    Read requirements from a text file and return them as a list.
    """
    requirements = []
    with open(File_path) as file_obj:
        requirements = file_obj.readlines()
    # Strip whitespace and newlines, and remove any empty strings
    requirements = [req.strip() for req in requirements if req.strip()]
    return requirements


with open("README.md", "r", encoding="UTF-8") as f:
    long_desc = f.read()


setup(
    name="Boston-Housing-Price-Prediction",
    version=__version__,
    author=Author_name,
    author_email=Author_email,
    description="ML app to predict house prices in Boston",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{Author_name}/{Repo_name}",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_packages(),
    python_requires=">=3.7",
)

# Run python setup.py install
