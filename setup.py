from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.2.0',
    description='Production-ready MLOps system for mental health text classification',
    author='Galkin_with_Skovorodnikov',
    license='MIT',
    entry_points={
        'console_scripts': [
            'study-mlops=src.cli:main',
        ],
    },
)
