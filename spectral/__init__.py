# Copyright (c) 2020, Vaishak Prasad
# See [LICENSE](https://gitlab.com/vaishakp/spectral/-/blob/main/LICENSE) file for details.
"""
Module for the analysis and handling of numerical relativity and gravitational waveform data.

Classes
-------
sim: Base class for NR simulations data.
                This is needed to retrieve and handle Numerical Relativity data.
spherical_array:    A 2D data-type.
                    Stores and manages two-dimensional data on surfaces of spherical topology.
modes_array: A data-type.
             Handle and work with mode coefficients.

"""
import os

package_directory = os.path.dirname(os.path.abspath(__file__))

print(package_directory)


def read_git_version():
    """Get the latest version number based on the last commit date,
    iff this is a git repo."""

    # print(package_directory)
    # Open the file
    vers = "-1"
    try:
        with open(
            package_directory + "/../public/version", "r", encoding="utf-8"
        ) as vers_file:
            vers = vers_file.readlines()[0]
    except Exception as excep:
        print(
            excep,
            "\n",
            "This is not a git repo! please use the version attribute instead!",
        )
    # with open(package_directory + "/../public/date.txt", "r") as vers_file:
    # vers = vers_file.read()[:10]

    return vers


__version__ = "2024.12.20"
