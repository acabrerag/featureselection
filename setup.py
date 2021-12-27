from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
required = []
EGG_MARK = '#egg='
for line in requirements:
    if line.startswith('git+'):
        if EGG_MARK in line:
            package_name = line[line.find(EGG_MARK) + len(EGG_MARK):]
            required.append("{0} @ {1}".format(package_name, line[0:line.find(EGG_MARK)]))

    else:
        required.append(line)

setup(
    name='featureselection',
    version='0.0.3',
    author='',
    author_email='',
    description='Library for machine learning models',
    packages=find_packages('module'),
    url='',
    license='LICENSE',
    include_package_data=True,
    install_requires=required
)
