from setuptools import setup, find_packages

setup(
    name="voice-of-client",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines()
    ],
    author="Your Organization",
    description="Voice Of Client AI Core",
    python_requires=">=3.9",
)
