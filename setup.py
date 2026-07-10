from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries :: Python Modules',
]

setup(
    name='hybridgbrf',
    version='1.0.0',
    description='Gradient Boosted Random Forest - A hybrid ML algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abhi1628/hybridgbrf',
    author='Abhishek Singh',
    author_email='abhisheksingh.cs@global.org.in',
    license='MIT',
    classifiers=classifiers,
    keywords='machine learning gradient boosting random forest hybrid ensemble',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        'dev': ['pytest>=6.0', 'pytest-cov', 'flake8'],
    },
    include_package_data=True,
    zip_safe=False,
)
