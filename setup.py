
"""pdfslash setup file."""

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('VERSION') as f:
    version = f.read().strip()


setup(
    name='pdfslash',
    version=version,
    url='https://github.com/openandclose/pdfslash',
    license='MIT',
    author='Open Close',
    author_email='openandclose23@gmail.com',
    description='Crop pdf margins from interactive interpreter.',
    long_description=readme,
    classifiers=[
        'Development Status :: 4 - Beta',
        "Intended Audience :: End Users/Desktop",
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    keywords='pdf crop trim MuPDF PyMuPDF briss',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pdfslash = pdfslash.slash:main',
        ],
    },
    python_requires='~=3.6',
    zip_safe=False,
)
