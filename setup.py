from setuptools import setup, find_packages

setup(
    name="FastSpike",
    version="0.0.1",
    description="Spiking neural networks for ML in Python",
    license="MIT License",
    url="https://github.com/mahbodnr/FastSpike",
    author="Mahbod Nouri",
    author_email="MahbodNouri@gmail.com",
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "torch"
    ],
)
