import numpy as np


# @njit(parallel=True)
def compute_fft(udata_x, delta_x):
    """Find the FFT of the samples in time-space,
    and return with the frequencies.

    Parameters
    ----------
    udata_x : 1d array
              The samples in time-space.

    delta_x : float
              The stepping delta_x

    Returns
    -------
    freqs :	1d array
            The frequency axis, shifted approriately.
    utilde : 1d array
             The samples in frequency space, with conventions applied.
    """

    # import necessary libraries.
    from numpy.fft import fft

    # FFT
    utilde_orig = fft(udata_x)

    # Apply conventions.
    utilde = set_fft_conven(utilde_orig)

    # Get frequency axes.
    Nlen = len(utilde)
    # message(Nlen)
    # Naxis			= np.arange(Nlen)
    # freq_orig		= fftfreq(Nlen)
    # freq_axis		= fftshift(freq_orig)*Nlen
    # delta_x		 = xdata[1] - xdata[0]

    # Naxis			 = np.arange(Nlen)
    # freq_axis = np.linspace(-0.5 / delta_x, 0.5 / delta_x, Nlen)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(Nlen, delta_x))

    return freq_axis, utilde


# @njit(parallel=True)
def compute_ifft(utilde, delta_f):
    """Find the inverse FFT of the samples in frequency-space,
    and return with the time axis.

    Parameters
    ----------
    utilde : 1d array
             The samples in frequency-space.

    delta_f : float
              The frequency stepping

    Returns
    -------
    time_axis : 1d array
                The time axis.

    udata_time : 1d array
                 The samples in time domain.
    """

    # import necessary libraries.
    from numpy.fft import ifft

    # FFT
    utilde_orig = unset_fft_conven(utilde)

    # Inverse transform
    udata_time = ifft(utilde_orig)

    # Get frequency axes.
    Nlen = len(udata_time)
    # message(Nlen)
    # Naxis			= np.arange(Nlen)
    # freq_orig		= fftfreq(Nlen)
    # freq_axis		= fftshift(freq_orig)*Nlen
    # delta_x		 = xdata[1] - xdata[0]

    # Naxis			 = np.arange(Nlen)
    delta_t = round(1.0 / (delta_f * Nlen), 4)
    # delta_t = 1.0 / (delta_f * Nlen)
    # Dt				= Nlen * delta_f/2

    time_axis = np.linspace(0, delta_t * Nlen, Nlen)
    # time_axis = np.arange(0, delta_t * Nlen, 1 / Nlen)

    return time_axis, udata_time


# @njit(parallel=True)
def set_fft_conven(utilde_orig):
    """Make a numppy fft consistent with the chosen conventions.
    This takes care of the zero mode factor and array position.
    Also, it shifts the negative frequencies using numpy's fftshift.

    Parameters
    ----------
    utilde_orig : 1d array
                  The result of a numpy fft.

    Returns
    -------
    utilde_conven :	1d array
                    The fft with set conventions.
    """

    # Multiply by 2, take conjugate.
    utilde_conven = 2 * np.conj(utilde_orig) / len(utilde_orig)
    # Restore the zero mode.
    utilde_conven[0] = utilde_conven[0] / 2
    # Shift the frequency axis.
    utilde_conven = np.fft.fftshift(utilde_conven)

    return utilde_conven


# @njit(parallel=True)
def unset_fft_conven(utilde_conven):
    """Make an actual conventional fft
    consistent with numpy's conventions.
    The inverse of set_conv.


    Parameters
    ----------
    utilde_conven : 1d array
                    The conventional fft data vector.

    Returns
    -------
    utilde_np : 1darray
                The fft data vector in numpy conventions.
    """

    utilde_np = np.fft.ifftshift(utilde_conven)

    utilde_np = len(utilde_np) * np.conj(utilde_np) / 2
    # message(utilde_original[0])
    utilde_np[0] *= 2
    # message(utilde_original[0])

    return utilde_np
