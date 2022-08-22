from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="dhg",
    version="0.9.0",
    author="Yifan Feng",
    author_email="evanfeng97@gmail.com",
    description="DHG is a Deep Learning Framework for Graph Neural Network and Hypergraph Neural Networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://deephypergraph.com/",
    packages=find_packages(),
    install_requires=["torch>=1.11.0", "optuna>=1.10.0", "numpy", "matplotlib", "requests", "sklearn",],
    test_requires=["pytest", "pytest-cov",],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
    ],
)

