# coding: utf-8

from setuptools import setup, find_packages



setup(
    name='helpful_iterators',
    version='0.3a',
    description='Helpful iterators',
    long_description='Helpful iterators for for utility tasks',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Topic :: Data Processing :: Iterators :: Utility',
    ],
    keywords='helpful iterators for for utility tasks',
    url='',
    author='d.grigoriev',
    author_email='d.grigoriev@spn.ru',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'cloudpickle',
        'dateparser',
        'langdetect',
        'more-itertools',
        'python-dateutil',
        'pytz',
        'regex',
        'six',
        'tzlocal',
        'jsonschema'
    ],
    include_package_data=True,
    zip_safe=False
)
