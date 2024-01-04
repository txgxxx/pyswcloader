from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='pyswcloader',
    version='1.1.1',
    description='analysis tool for neuron swc files',
    long_description=long_description,
    # include_package_data=True,
    package_data={
        'pyswcloader': ['database/*',]
        },
    author='tsgao',
    author_email='gaots@ion.ac.cn',
    url='https://github.com/txgxxx/pyswcloader/',
    license='MIT',
    keywords='swc',
    packages=['pyswcloader'],
    py_modules=['swc', 'brain', 'projection', 'projection_batch', 'distance', 'cluster', 'visualization'],
    install_requires=[
        'pynrrd', 
        'numpy', 
        'pandas', 
        'treelib', 
        'tqdm', 
        'matplotlib', 
        'vispy', 
        'seaborn', 
        'kaleido', 
        'glob2', 
        'scikit-learn', 
        'statistics', 
        'scipy',
        'rdp'
        ]
)