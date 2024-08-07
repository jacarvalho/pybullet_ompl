from setuptools import setup, find_packages
from codecs import open
from os import path


from pb_ompl import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='pb_ompl',
      version=__version__,
      description='Pybullet OMPL Interface',
      # author='Joao Carvalho',
      # author_email='joao@robots-learning.de',
      packages=find_packages(where=''),
      install_requires=requires_list,
      )
