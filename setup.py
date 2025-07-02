from setuptools import setup, find_packages

setup(
    name='ecoperceiver',
    version='0.2.1',
    author='Matthew Fortier',
    author_email='fortier.matt@gmail.com',
    description='A multimodal model and dataloader for carbon flux modelling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mjfortier/EcoPerceiver',
    packages=find_packages(),
    install_requires=[
        'numpy==1.24.3',
        'pandas==2.0.1',
        'torch==2.0',
        'torchvision=0.16.2',
        'einops==0.6.1'
    ],
    python_requires='>=3.9',
)
