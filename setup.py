from setuptools import setup,find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='pyswcloader',
    version='1.1.1',
    description='analysis tool for neuron swc files',
    long_description=long_description,
    # include_package_data=True,
    package_data={
        'pyswcloader': ['database/*', 'web_summary/templates/*']
        },
    author='tsgao',
    author_email='gaots@ion.ac.cn',
    url='https://github.com/txgxxx/pyswcloader/',
    license='MIT',
    keywords='swc',
    # packages=['pyswcloader', 'pyswcloader.reader', 'pyswcloader.projection', 'pyswcloader.visualization', 'pyswcloader.web_summary'],
    packages=find_packages(include='pyswcloader*', exclude='pyswcloader.database'),
    py_modules=['distance', 'cluster', 'summary'],
    install_requires=[
        'pynrrd', 
        'numpy==1.26.4',
        'pandas==1.4.3',
        'treelib', 
        'tqdm', 
        'matplotlib==3.5.0',
        'vispy', 
        'seaborn==0.13.2',
        'kaleido', 
        'glob2', 
        'scikit-learn', 
        'statistics', 
        'scipy',
        'rdp',
        "Jinja2",
        "glfw",
        "pyglet",
        "plotly",
        "distinctipy",
        "pillow",
        ]
)