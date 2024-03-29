import setuptools  # type: ignore

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vmfpy",
    version="0.9.4",
    author="Lassi Säike",
    description="A Valve Map Format (VMF) and Valve Material Type (VMT) parser.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lasa01/vmfpy",
    packages=["vmfpy"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3"
    ],
    keywords="vmf valve parser",
    install_requires=["vdf", "vpk"],
)
