import codecs
import os
import re
from os.path import exists, join

from setuptools import find_packages, setup


def find_version(*file_paths: str) -> str:
    with codecs.open(join(*file_paths), "r") as fp:
        version_file = fp.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


if exists("README.md"):
    with open("README.md", encoding="utf8") as fh:
        LONG_DESC = LONG_DESC = fh.read()
else:
    LONG_DESC = ""


def strip_comments(line):
    if "find" in line:
        return ""
    return line.split("#", 1)[0].strip()


def reqs(*f):
    with open(os.path.join(os.getcwd(), *f), encoding="utf-8") as f:
        return [strip_comments(L) for L in f if strip_comments(L)]


install_requires = reqs("requirements.txt")


print(install_requires)

setup(
    name="spk_emb_ja",
    version=find_version("src", "__init__.py"),
    description="spk embedding trained by ja spk dataset",
    author="Kai Washizaki",
    author_email="bandad.kw@gmail.com",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    package_data={"": ["_example_data/*"]},
    packages=find_packages(include=["src*", "tests*"]),
    include_package_data=True,
    install_requires=install_requires,
)