#! /usr/bin/env python

# Author: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause

import os


def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    # Avoid non-useful msg:
    # "Ignoring attempt to set 'name' (from ... "
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)

    config.add_subpackage('skgp')

    return config


def setup_package():
    metadata = dict(
        name="skgp",
        author="Jan Hendrik Metzen",
        author_email="jhm@informatik.uni-bremen.de",
        description="Extended Gaussian Process functionality for sklearn",
        long_description=open("README.rst").read(),
        license="new BSD",
        url="https://github.com/jmetzen/skgp",
        version="0.0",
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Programming Language :: Python",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.3",
        ],
        requires=["numpy", "scipy", "sklearn"])

    metadata['configuration'] = configuration

    from numpy.distutils.core import setup

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
