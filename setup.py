import os
from setuptools import setup, find_packages

# package description and keywords
description = ('Python-based tools for correcting altimetry data for '
    'surface mass balance and firn processes')
keywords = 'Surface Mass Balance, firn height, ICESat, ICESat-2, Operation IceBridge'
# get long_description from README.rst
with open("README.rst", "r") as fh:
    long_description = fh.read()
long_description_content_type = "text/x-rst"

# get version
with open('version.txt') as fh:
    version = fh.read()

# list of all scripts to be included with package
scripts=[os.path.join('scripts',f) for f in os.listdir('scripts') if f.endswith('.py')]

# install requirements and dependencies
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
    dependency_links = []
else:
    # get install requirements
    with open('requirements.txt') as fh:
        install_requires = [line.split().pop(0) for line in fh.read().splitlines()]
    # dependency links
    dependency_links = ['https://github.com/SmithB/pointCollection/tarball/master']

setup(
    name='SMBcorr',
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    url='https://github.com/tsutterley/SMBcorr',
    author='Tyler Sutterley',
    author_email='tsutterl@uw.edu',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    keywords=keywords,
    packages=find_packages(),
    install_requires=install_requires,
    dependency_links=dependency_links,
    scripts=scripts,
    include_package_data=True,
)
