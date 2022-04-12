import os
import io
from setuptools import setup, find_packages, Command

required_python = ">=3.0"
URL = "http://water.engr.psu.edu/shen/"

cwd = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(cwd, "requirements.txt"), encoding="utf-8") as f:
        required = f.read().split("\n")
except:
    required = []

# import READ.MD
try:
    with io.open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = None

setup(
    name="hydroDL",
    version="0.1.0",
    description="Hydrological Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MHPI",
    author_email="cshen@engr.psu.edu",
    python_requires=required_python,
    url=URL,
    packages=["hydroDL"],
    install_requires=required,
    include_package_data=True,
    license="Non-Commercial Software License",

)

