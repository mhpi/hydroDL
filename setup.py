"""hydroDL package.

"""

from setuptools import setup
import re

# To use a consistent encoding
from codecs import open
import os

here = os.path.abspath(os.path.dirname(__file__))

verstr = "unknown"
VERSIONFILE = "hydroDL/_version.py"
with open(VERSIONFILE, "r") as f:
    verstrline = f.read().strip()
    pattern = re.compile(r"__version__ = ['\"](.*)['\"]")
    mo = pattern.search(verstrline)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    long_description_content_type = "text/markdown"


REQUIREMENTS = [
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "statsmodels",
    "matplotlib",
    "json",
    # "psycopg2",
]

TEST_REQUIREMENTS = [
    "pytest",
    #  'coveralls',
    #  'pytest-cov',
    #  'pytest-mpl'
]

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development ",
    # 'License :: OSI Approved :: MIT License',
    "Operating System :: OS Independent",
    "Programming Language :: Python",
]


setup(
    name="hydroDL",
    version=verstr,
    description="",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url="https://github.com/mhpi/hydroDL",
    author="Dapeng Feng (PhD Student, Penn State) and Kuai Fang (PhD., Penn State)",
    author_email="geofkwai@gmail.com",
    classifiers=CLASSIFIERS,
    keywords="hydrology deeplearning ML hybrid model",
    packages=[
        "hydroDL",
        "hydroDL.data",
        "hydroDL.master",
        "hydroDL.model",
        "hydroDL.post",
        "hydroDL.utils",
    ],
    install_requires=REQUIREMENTS,
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files
    #
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],  # Optional
    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # `pip` to create the appropriate form of executable for the target
    # platform.
    #
    # For example, the following would provide a command called `sample` which
    # executes the function `main` from this package when invoked:
    # entry_points={  # Optional
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    project_urls={  # Optional
        "Bug Reports": "https://github.com/mhpi/hydroDL/issues",
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        "Source": "https://github.com/mhpi/hydroDL",
    },
)
