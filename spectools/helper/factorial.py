"""Memoized exact factorial.

The spin-weighted spherical harmonic routines evaluate factorials of the
same handful of integers repeatedly, inside loops over
:math:`(\\ell, m)`. This module keeps a process-wide cache so each value is
computed once.

Results are exact Python integers, not floats, which matters for the
arbitrary-precision harmonic paths in :mod:`spectools.spherical`.
"""

#: Module-level cache of already-computed factorials, keyed by integer.
#: Grows for the lifetime of the process.
fact_dict = {0: 1, 1: 1}


def factorial(number):
    """Return ``number!`` exactly, caching the result.

    Parameters
    ----------
    number : int
        A non-negative integer.

    Returns
    -------
    int
        The exact factorial of ``number``.

    Notes
    -----
    Recursive, and memoized in :data:`fact_dict`, so every intermediate
    factorial below ``number`` is cached as a side effect. A negative
    argument recurses without a base case and will raise
    ``RecursionError``.

    Examples
    --------
    >>> from spectools.helper.factorial import factorial
    >>> factorial(5)
    120
    """
    if number not in fact_dict.keys():
        fact_dict.update({number: factorial(number - 1) * number})

    return fact_dict[number]
