from setuptools import setup, find_packages

setup(
    name="cdbnn",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "pydicom",
        "astropy",
        "Pillow",
        "pandas"
    ],
    entry_points={
        "console_scripts": [
            "cdbnn=main:main"
        ]
    }
)
