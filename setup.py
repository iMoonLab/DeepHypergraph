import setuptools

with open('README.md', "r") as f:
    long_description = f.read()

setuptools.setup(
    name='SuperMoon',
    version='0.0.1',
    author='yifanfeng',
    author_email='evanfeng97@gmail.com',
    description="Hypergraph Toolbox",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/imoonlab/THU-SuperMoon',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
