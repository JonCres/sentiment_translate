from setuptools import setup, find_packages

setup(
    name="ai-core-template",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines()
    ],
    author="Your Organization",
    description="Reusable AI Core Template with Prefect",
    python_requires=">=3.9",
)
