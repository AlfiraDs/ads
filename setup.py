from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='ads',
      version='0.1',
      description='ML helper code',
      author='Aliaksandr Laskov',
      license='All rights reserved',
      packages=find_packages(),
      zip_safe=False,
      install_requires=required,
      dependency_links=[]
      )
