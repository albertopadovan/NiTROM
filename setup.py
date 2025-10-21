from setuptools import setup, find_packages

setup(
    name="NiTROM_GPU",
    version="0.1.0",
    packages=find_packages(include=["NiTROM_GPU", "NiTROM_GPU.*"]),
    install_requires=["numpy", "scipy", "numba", "pymanopt", "matplotlib", "torch"],
    description="Non-intrusive Trajectory-based Reduced-Order Modelling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alberto Padovan",
    author_email="padovan3@illinois.edu",
    url="https://github.com/albertopadovan/NiTROM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
