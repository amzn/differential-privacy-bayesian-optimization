import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='dpareto',
    version='0.1.3',
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
        'tensorflow==1.15',
        'gpflow @ git+https://github.com/GPflow/GPflow.git@ce5ad7ea75687fb0bf178b25f62855fc861eb10f',
        'gpflowopt @ git+https://github.com/GPflow/GPflowOpt.git',
        'seaborn',
        'psutil',
        'mxnet',
        'autodp',
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
