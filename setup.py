from setuptools import setup, find_packages

setup(
    name='geodesic_interpolate',
    version='0.0.1',
    author='Louie Slocombe',
    author_email='louies@hotmail.co.uk',
    description='Interpolation and smoothing of reaction paths with geodesics in '
                'redundant internal coordinates.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LouieSlocombe/geodesic_interpolate',
    packages=find_packages(include=['geodesic_interpolate', 'geodesic_interpolate.*']),
    package_data={
        'geodesic_interpolate': [
            'data/*',
        ],
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
    install_requires=[
        'numpy',
        'scipy',
        'ase',
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ],
    },
)
