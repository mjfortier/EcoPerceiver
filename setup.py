from setuptools import setup, find_packages

setup(
    name='ecoperceiver',
    version='0.1.0',
    author='Matthew Fortier',
    author_email='fortier.matt@gmail.com',
    description='A multimodal model and dataloader for carbon flux modelling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mjfortier/EcoPerceiver',
    packages=find_packages(),
    install_requires=[
    ],
    python_requires='>=3.8',
)
