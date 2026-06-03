from pathlib import Path
import os

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py as _build_py
from Cython.Build import cythonize

try:
    import numpy as np
    include_dirs = [np.get_include()]
except Exception:
    include_dirs = []


PACKAGE = os.environ.get("PACKAGE_NAME", "spectools")
VERSION = os.environ.get("PACKAGE_VERSION", "0.0.0+private")


def module_name_from_path(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


py_files = [
    p for p in Path(PACKAGE).rglob("*.py")
    if p.name != "__init__.py"
    and "tests" not in p.parts
]


extensions = [
    Extension(
        module_name_from_path(p),
        [str(p)],
        include_dirs=include_dirs,
    )
    for p in py_files
]


class build_py(_build_py):
    """Copy only __init__.py files, not implementation .py files.

    Keep __init__.py minimal. If sensitive logic is in __init__.py,
    move it into normal modules first.
    """

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [
            (pkg, mod, file)
            for (pkg, mod, file) in modules
            if mod == "__init__"
        ]


setup(
    name=PACKAGE,
    version=VERSION,
    packages=find_packages(exclude=("tests", "tests.*")),
    ext_modules=cythonize(
        extensions,
        build_dir="build/cythonized",
        compiler_directives={
            "language_level": "3",
            "embedsignature": False,
            "binding": False,
        },
        annotate=False,
    ),
    cmdclass={"build_py": build_py},
    zip_safe=False,
)
