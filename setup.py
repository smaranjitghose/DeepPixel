from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1'
PACKAGE_NAME = 'DeepPixel'
AUTHOR = 'Smaranjit Ghose'
AUTHOR_EMAIL = 'smaranjitghose@protonmail.com'
URL = 'https://github.com/smaranjitghose/deepixel'
KEYWORDS = "deep-pixel deeplearing computervision iqa explainable-ai"


LICENSE = 'MIT License'
DESCRIPTION = 'An open source package to plug and play common computer vision and image processing tasks using deep learning under the hood'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ['opencv-python', 'imutils', 'scikit-image','Pillow', 'argparse', 'tensorflow']

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      keywords=KEYWORDS,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
