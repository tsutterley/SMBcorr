from setuptools import setup, find_packages
setup(
    name='SMBcorr',
    version='0.0.0.1',
    description='Python-based tools for correcting altimetry data for surface mass balance and firn processes',
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
    keywords='Surface Mass Balance, firn height, ICESat, ICESat-2, Operation IceBridge',
    packages=find_packages(),
    install_requires=['numpy','scipy','pyproj','netCDF4','scikit-learn'],
)
