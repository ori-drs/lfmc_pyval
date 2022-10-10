from setuptools import setup, find_packages

setup(
    name='lfmc_pyval',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/ori-drs/lfmc_pyval',
    author='Siddhant Gangapurwala',
    author_email='siddhant@robots.ox.ac.uk',
    python_requires='>=3.6.0,<3.10.0',
    install_requires=[
        'pyyaml',
        'numpy',
        'pybullet',
        'torch'
    ]
)
