# author: Matt Clifford <matt.clifford@bristol.ac.uk> <matt.clifford@bristol.ac.uk>
from setuptools import setup, find_packages

def dependencies_from_file(file_path):
    required = []
    with open(file_path) as f:
        for l in f.readlines():
            l_c = l.strip()
            # get not empty lines and ones that do not start with python
            # comment "#" (preceded by any number of white spaces)
            if l_c and not l_c.startswith('#'):
                required.append(l_c)
    return required

def get_long_description():
    with open('README.md', encoding="utf-8") as f:
        text = f.read()
    return text


setup(name='CLIME',
      version='0.1',
      packages=find_packages(),
      install_requires=dependencies_from_file('requirements.txt'),
      python_requires='>=3.9',
      include_package_data=True,
      author="Matt Clifford",
      author_email="matt.clifford@bristol.ac.uk",
      description="CLIME: Class balanced Local Interpretable Model-agnostic Explanations.",
      long_description=get_long_description(),
      long_description_content_type='text/markdown',
      )
