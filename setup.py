from setuptools import setup, find_packages
setup(
    name='rag_utilities',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)