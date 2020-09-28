import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='dpareto',
    version='0.1.5',
    license='Apache 2.0',
    description='Automatic Discovery of Privacy-Utility Pareto Fronts',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Amazon',
    author_email='tdiethe@amazon.com',
    url='https://github.com/amzn/differential-privacy-bayesian-optimization',
    download_url='https://github.com/amzn/differential-privacy-bayesian-optimization/archive/v_01.tar.gz',
    keywords=['Differential privacy', 'Bayesian optimization'],
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy==1.16.1',
        'tensorflow==1.15.4',
        'gpflow @ git+https://github.com/GPflow/GPflow.git@ce5ad7ea75687fb0bf178b25f62855fc861eb10f',
        'gpflowopt @ git+https://github.com/GPflow/GPflowOpt.git@f1c268e6b5dc4d7f458e06c59095901d55b73c32',
        'seaborn==0.9.1',
        'psutil==5.6.6',
        'mxnet==1.5.1',
        'autodp==0.1',
      ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
