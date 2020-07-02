#!/usr/bin/env python3

from setuptools import find_packages, setup


# Long description
with open("README.md", encoding="utf-8") as f:
    readme = f.read()


# Requirements
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [x for x in map(str.strip, f.read().splitlines())
                    if x and not x.startswith("#")]


setup(
    name='spynet',
    version='1.0',
    author='Guillem Orellana Trullols',
    author_email='guillem.orellana@gmail.com',
    url='https://github.com/Guillem96/spynet-pytorch',
    description='Compute optical flow with a lightweighted NN',
    long_description=readme,
    keywords='optical-flow video pytorch deep-learning',
    install_requires=requirements,
    packages=find_packages(),
    zip_safe=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)