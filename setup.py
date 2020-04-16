from setuptools import setup

setup(name='deeppixel',
      version='0.0.1',
      description='An open source package to plug and play common computer vision and image processing tasks using deep learning under the hood',
      long_description='An open source package to plug and play common computer vision and image processing tasks using deep learning under the hood',
      url='https://github.com/smaranjitghose/deeppixel.git',
      author='Smaranjit Ghose',
      author_email='smaranjitghose@protonmail.com',
      license='MIT',
      packages=['deeppixel'],
      include_package_data=True,
      zip_safe=False,
      install_requires=['opencv-python', 'imutils', 'scikit-image',
                        'Pillow', 'argparse', 'tensorflow']
      )
