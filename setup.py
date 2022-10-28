from setuptools import setup, find_packages

setup_requires = []

install_requires = [
    'loguru~=0.6.0',
    'transformers~=4.12.5',
    'pyyaml~=6.0',
    'tqdm~=4.55.0',
    'nltk~=3.4.5',
    'requests~=2.25.1',
]

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Human Machine Interfaces"
]

# with open("README.md", "r", encoding="utf-8") as f:
#     long_description = f.read()

setup(
    name='simtester',
    version='0.1.0',
    author='shuyu guo',
    author_email='guoshuyu225@gmail.com',
    description='An Open-Source Toolkit for Evaluating User Simulator of Task-oriented Dialogue System',
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # url='https://github.com/RUCAIBox/CRSLab',
    packages=[
        package for package in find_packages()
        if package.startswith('simtester')
    ],
    classifiers=classifiers,
    install_requires=install_requires,
    setup_requires=setup_requires,
    python_requires='>=3.6',
)
