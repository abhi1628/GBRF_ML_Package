from setuptools import setup, find_packages, setuptools

classifiers = [
  'Development Status :: 5 - Production/Stable',
  'Intended Audience :: Education',
  'Operating System :: OS Independent',
  'License :: OSI Approved :: MIT License',
  'Programming Language :: Python :: 3',
  "Programming Language :: Python :: 3.7",
  "Programming Language :: Python :: 3.8",
  "Programming Language :: Python :: 3.9",
]

setup(
  name='gbrf',
  version='1.0.7',
  description='Gradient Boosted Random Forest Classifier (gbrf)',
  long_description_content_type='text/markdown',
  long_description=open('README.txt').read() + '\n\n' + open('CHANGELOG.txt').read(),
  url='',
  author='Abhishek Singh',
  author_email='abhishek.ec@global.org.in',
  license='MIT',
  classifier=classifiers,
  keywords='machine learning',
  packages=setuptools.find_packages(),
  install_requires=["numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.1.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.4",]
)
