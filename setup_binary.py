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


# Paths relative to the repo root.
SKIP_FILES = {
    "spectools/spherical/Yslm_full_vec.py",
    # Add more problematic files here:
    # "spectools/some/problematic_module.py",
}


SKIP_DIRS = {
    "tests",
    "__pycache__",
}


def module_name_from_path(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def should_skip(path: Path) -> bool:
    path_str = path.as_posix()

    if path_str in SKIP_FILES:
        return True

    if any(part in SKIP_DIRS for part in path.parts):
        return True

    if path.name == "__init__.py":
        return True

    return False


py_files = [
    p for p in Path(PACKAGE).rglob("*.py")
    if not should_skip(p)
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
    """
    Copy only __init__.py files into the wheel.

    This prevents skipped implementation .py files from being copied as source.
    Therefore skipped modules will not be importable unless you provide
    compiled/stub/pyc replacements separately.
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
        force=True,
    ),
    cmdclass={"build_py": build_py},
    zip_safe=False,
)