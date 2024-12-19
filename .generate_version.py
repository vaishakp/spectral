#!/usr/bin/env python

import os
from datetime import date

source_directory = os.path.dirname(os.path.abspath(__file__))

# print(source_directory)


def write_git_version():
    """Write the latest version number based on the last commit date
    to the public folder and to __init__.py file"""

    # print(package_directory)
    # Open the file
    # Get the version
    # vers = os.popen(f'git -C {source_directory} log -1 --date=short | grep Date').read()[8:-1]
    vers = str(date.today())
    vers = vers.replace("-", ".")

    # Write to public/version
    with open(source_directory + "/public/version", "w") as vers_file:
        vers_file.writelines([vers])

    # write to __init__.py
    ee = os.popen(
        f"sed -i '/__version__/c\\__version__ = \"{vers}\"' {source_directory}/spectral/__init__.py"
    )
    # with open(package_directory+'/spectral/__init__.py', 'r') as init_file:
    # with open(package_directory + "/../public/date.txt", "r") as vers_file:
    # vers = vers_file.read()[:10]

    ee = os.popen(
        f"sed -i '/version/c\\ \tversion=\"{vers}\",' {source_directory}/setup.py"
    )
    print("Version", vers)

    # Generate badge
    badge = f"""<svg width="140" height="20" xmlns="http://www.w3.org/2000/svg">
                <rect width="80" height="20" rx="0" ry="5" fill="grey" x="0" y="0" />
                <rect width="60" height="20" fill="blue" x="80" y="0" />
                <text x="6" y="13" fill="white" font-size="11">pypi package</text>
                <text x="83" y="14" fill="white" font-size="11">{vers}</text>
                </svg>
            """

    with open("docs/vers_badge.svg", "w") as svgf:
        svgf.write(badge)

    return vers


write_git_version()
