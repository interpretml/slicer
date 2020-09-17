import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slicer",
    version="0.0.3",
    author="Anonymous developers from identifiable interpretability packages.",
    author_email="interpret@microsoft.com",
    description="A small package for big slicing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/interpretml/slicer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
