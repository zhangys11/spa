# from distutils.core import setup
from setuptools import setup

setup(
    # Application name:
    name="spa-ds",

    # Version number:
    version="2.0.7",

    # Application author details:
    author="Yinsheng Zhang (Ph.D.)",
    author_email="oo@zju.edu.cn",

    # Packages
    packages=["spa", "spa.fs", "spa.fs.glasso", "spa.gui", "spa.gui.templates", "spa.dr", "spa.cla",
              "spa.vis", "spa.io", "spa.data", "spa.io.aug", "spa.mh"],

    # package_dir={'': 'spa'},
    # package_dir={'spa.dr': 'src/spa/dr', 'spa.cla': 'src/spa/cla', 'spa.vis': 'src/spa/vis'},

    # Include additional files into the package
    include_package_data=True,

    # Details
    url="http://pypi.python.org/pypi/spa_ds/",

    #
    license="LICENSE.txt",
    description="Data science toolkit for spectroscopic profiling data analysis.",

    long_description_content_type='text/markdown',
    long_description=open('README.md', encoding='utf-8').read(),

    # Dependent packages (distributions)
    install_requires=[
        "flask",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "PyWavelets",
        "seaborn",
        "joblib",
        "tqdm",
        "h5py",
        "pyNNRW",
        "cla",
        "pyDRMetrics",
        "wDRMetrics",
        # "pyMFDR", # avoid importing keras and tf stuffs unless needed
        "cs1",
        "ctgan",
        "cvxpy",
        "asgl",
        "xgboost",
    ],

    package_data={
        "": ["*.txt", "*.csv", "*.png", "*.jpg", "*.json"],
    }
)

# To Build and Publish (for developer only),
# Run: python -m build --wheel
# Run: python -m pyc_wheel spa_ds.whl  [optional]
# or
# Run: python setup.py sdist bdist_wheel; twine upload dist/*
