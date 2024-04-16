from waveformtools.waveformtools import message
import numpy as np
from waveformtools.integrate import TwoDIntegral
from waveformtools.single_mode import SingleMode
from spectral.spherical.swsh import Yslm_vec

def CheckRegReq(data):
    """Check if a function requires regularization.

    Parameters
    ----------
    data : 1d array
           A 1d array of the data to check.

    Returns
    -------
    check_reg : list
                a list containg the list of boundary points where
                regularization may be required.
    """
    nlen = len(data)
    nhlen = int(nlen / 2)
    nrlen = nlen - nhlen

    first_half = data[:nhlen]
    second_half = data[nhlen:]

    check_reg = [0, 0]

    toln = int(nlen / 10)
    if np.argmax(np.absolute(first_half)) <= toln:  # Added tolerence Apr 8 2023
        check_reg[0] = 1

    if np.argmax(np.absolute(second_half)) >= nrlen - toln:  # Here as well
        check_reg[1] = 1

    # if 1 in check_reg:
    # print('Reqularization required at', check_reg)

    return check_reg


def SHExpand(
    func,
    info,
    method_info,
    err_info=False,
    auto_ell_max=False,
    res_tol_percent=3,
    reg=False,
    reg_order=1,
):
    """Expand a given function in spin weight 0 spherical harmonics
    upto an optimal :math:`\\ell \\leq \\ell_{max}`.

    Parameters
    ----------
    func : ndarray
           The function to be expanded.
    info : Grid
           An instance of the Spherical grid class
           that stores the details of the structure
           of a grid on a topological sphere.
    method_info : MethodInfo
                  An instance of the method info
                  class that contains informations
                  about the numerical methods
                  to be used during the following
                  operations.
    err_info : bool
               Whether or not to compute and return
               the error measures related to the
               SH representation.

    Returns
    -------
    modes : dict
            The modes as a dictionary whose keys are lm.
    """

    if info.grid_type == "GL":
        assert method_info.ell_max == info.L, (
            "The GL grid L must be same" " as ell_max of requested expansion"
        )

    if auto_ell_max:
        message(
            "Using SHExpandAuto: "
            " Will automatically find optimal "
            " ell_max",
            message_verbosity=2,
        )

        results = SHExpandAuto(
            func,
            info,
            method_info,
            err_info,
            res_tol_percent,
            reg,
            reg_order=reg_order,
        )

    else:
        message(
            "Using ShExpandSimple:"
            " Expanding upto user prescribed"
            f" ell_max {method_info.ell_max}",
            message_verbosity=2,
        )

        results = SHExpandSimple(
            func, info, method_info, err_info, reg=reg, reg_order=reg_order
        )

    return results


def SHRegularize(func, theta_grid, check_reg, order=1):
    """Regularize an SH expansion"""

    reg_func = func.copy()

    if bool(check_reg[0]):
        message("Regularizing north end ", message_verbosity=2)
        reg_func *= (theta_grid) ** order

    if bool(check_reg[1]):
        message("Regularizing south end ", message_verbosity=2)
        reg_func *= (theta_grid - np.pi) ** order

    return reg_func


def SHDeRegularize(func, theta_grid, check_reg, order=1):
    """Return the original funtion given the regularized functions"""

    orig_func = func.copy()

    if bool(check_reg[0]):
        orig_func /= (theta_grid) ** order

    if bool(check_reg[1]):
        orig_func /= (theta_grid - np.pi) ** order

    return orig_func


def SHExpandAuto(
    func,
    info,
    method_info,
    err_info=False,
    res_tol_percent=3,
    reg=False,
    reg_order=1,
    check_reg=None,
):
    """Expand a given function in spin weight 0 spherical harmonics
    upto an optimal :math:`\\ell \\leq \\ell_{max}` that is
    automatically found.

    Additionally, if requested, this routine can:

    1. regularize a function and expand and return the
       modes of the regularized function and the associated
       regularization details.
    2. Compute diagnostic information in terms of residue
       per mode.
    3. The RMS deviation of the reconstructed expansion from the
       original function.

    Parameters
    ----------
    func : ndarray
           The function to be expanded.
    info : Grid
           An instance of the Spherical grid class
           that stores the details of the structure
           of a grid on a topological sphere.
    method_info : MethodInfo
                  An instance of the method info
                  class that contains informations
                  about the numerical methods
                  to be used during the following
                  operations.
    err_info : bool
               Whether or not to compute and return
               the error measures related to the
               SH representation.
    check_reg : list, optional
                A list of two integers (0,1)
                that depicts whether or not to
                regularize the input function
                at the poles.
    Returns
    -------
    modes : dict
            The modes as a dictionary whose keys are lm.


    Notes
    -----
    When regularization is requested,
        1. To compute the total RMS deviation,
           the orginal form is used.
        2. To compute the rms deviation per mode,
           regularized expression is used.

    """

    #####################
    # Prepare
    #####################

    orig_func = func.copy()

    # from scipy.special import sph_harm
    import sys

    theta_grid, phi_grid = info.meshgrid

    ell_max = method_info.ell_max
    method = method_info.int_method

    # from waveformtools.single_mode import SingleMode

    modes = {}

    # if method != "GL":
    #    SinTheta = np.sin(theta_grid)
    # else:
    #    SinTheta = 1

    #####################

    ####################
    # Regularize
    ####################
    if reg:
        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message("Regularizing function", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    #####################

    #################
    # Zeroth residue
    #################

    recon_func = np.zeros(func.shape, dtype=np.complex128)

    # The first residue is the maximum residue
    # with zero as reconstructed function
    res1 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))

    # The list holding all residues
    all_res = [res1]

    #################

    #######################
    # Expand
    #######################

    for ell in range(ell_max + 1):
        emm_list = np.arange(-ell, ell + 1)

        emmCoeffs = {}

        for emm in emm_list:
            Ylm = Yslm_vec(
                spin_weight=0,
                emm=emm,
                ell=ell,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

            integrand = func * np.conjugate(Ylm)

            uu = np.isnan(integrand.any())

            if uu:
                raise ValueError("Nan found!")

            Clm = TwoDIntegral(integrand, info, method=method)

            recon_func += Clm * Ylm

            emmCoeffs.update({f"m{emm}": Clm})

        if ell % 2 == 0:
            res2 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))

            dres_percent = 100 * (res2 / res1 - 1)

            if dres_percent > res_tol_percent:
                all_res.append(res2)
                message(
                    f" ell_max residue increase error of {dres_percent} %",
                    message_verbosity=1,
                )

                ell_max = ell - 1
                message(
                    "Auto setting ell max to {ell_max} instead",
                    ell_max,
                    message_verbosity=1,
                )
                break

            else:
                res1 = res2
                all_res.append(res1)

        elif ell == ell_max:
            res2 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))
            all_res.append(res2)

        modes.update({f"l{ell}": emmCoeffs})

    ############################

    #################################
    # update details
    #################################

    result = SingleMode(modes_dict=modes)
    result._Grid = info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if err_info:
        from waveformtools.diagnostics import RMSerrs

        recon_func = SHContract(modes, info, ell_max)

        ################################
        # Compute total RMS deviation
        # of the expansion
        ###############################

        if reg:
            if np.array(check_reg).any() > 0:
                message(
                    "De-regularizing function" "for RMS deviation computation",
                    message_verbosity=2,
                )

                recon_func = SHDeRegularize(
                    recon_func, theta_grid, check_reg, order=reg_order
                )

        Rerr, Amin, Amax = RMSerrs(orig_func, recon_func, info)
        err_info_dict = {"RMS": Rerr, "Amin": Amin, "Amax": Amax}

        ############################
        # Update error details
        ############################

        result.error_info = err_info_dict
        result.residuals = all_res

        even_mode_nums = np.arange(0, ell_max, 2)

        residual_axis = [-1] + list(even_mode_nums)

        if ell_max % 2 == 1:
            residual_axis += [ell_max]

        result.residual_axis = residual_axis

        if Rerr > 0.1:
            message(
                f"Residue warning {Rerr}!  Inaccurate representation.",
                message_verbosity=0,
            )

    #####################################

    return result


def SHExpandSimple(
    func,
    info,
    method_info,
    err_info=False,
    reg=False,
    reg_order=1,
    check_reg=None,
):
    """Expand a given function in spin weight 0 spherical harmonics
    upto a user prescribed :math:`\\ell_{max}`.

    Additionally, if requested, this routine can:

    1. regularize a function and expand and return the
       modes of the regularized function and the associated
       regularization details.
    2. Compute diagnostic information in terms of residue
       per mode.
    3. The RMS deviation of the reconstructed expansion from the
       original function.


    Parameters
    ----------
    func : ndarray
           The function to be expanded.
    info : Grid
           An instance of the Spherical grid class
           that stores the details of the structure
           of a grid on a topological sphere.
    method_info : MethodInfo
                  An instance of the method info
                  class that contains informations
                  about the numerical methods
                  to be used during the following
                  operations.
    err_info : bool
               Whether or not to compute and return
               the error measures related to the
               SH representation.

    check_reg : list, optional
                A list of two integers (0,1)
                that depicts whether or not to
                regularize the input function
                at the poles.

    Returns
    -------
    modes : dict
            The modes as a dictionary whose keys are lm.

    Notes
    -----
    When regularization is requested,
        1. To compute the total RMS deviation,
           the orginal form is used.
        2. To compute the rms deviation per mode,
           regularized expression is used.


    """
    # from scipy.special import sph_harm
    import sys

    # from waveformtools.single_mode import SingleMode

    orig_func = func.copy()

    theta_grid, phi_grid = info.meshgrid

    ell_max = method_info.ell_max

    method = method_info.int_method

    message(
        f"SHExpandSimple: expansion ell max is {ell_max}", message_verbosity=3
    )

    # Good old Modes dict
    # modes = {}

    # if method != "GL":
    #    SinTheta = np.sin(theta_grid)
    # else:
    #    SinTheta = 1

    if reg:
        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message("Regularizing function", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    result = SingleMode(ell_max=ell_max)

    recon_func = np.zeros(func.shape, dtype=np.complex128)

    res1 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))

    all_res = [res1]

    for ell in range(ell_max + 1):
        emm_list = np.arange(-ell, ell + 1)

        # Subdict of modes
        # emmCoeffs = {}

        for emm in emm_list:
            Ylm = Yslm_vec(
                spin_weight=0,
                emm=emm,
                ell=ell,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

            integrand = func * np.conjugate(Ylm)

            uu = np.isnan(integrand.any())

            # print(uu)
            if uu:
                raise ValueError("Nan found!")

            Clm = TwoDIntegral(integrand, info, method=method)

            recon_func += Clm * Ylm

            # emmCoeffs.update({f"m{emm}": Clm})
            # print(Clm)
            # message("Clm ", Clm, message_verbosity=2)

            result.set_mode_data(ell, emm, Clm)

        res = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))
        all_res.append(res)

        # modes.update({f"l{ell}": emmCoeffs})

    # result2 = SingleMode(modes_dict=modes)

    # message(f"result2 ell max {result2.ell_max}", message_verbosity=1)

    result._Grid = info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if err_info:
        from waveformtools.diagnostics import RMSerrs

        recon_func = SHContract(result, info, ell_max)

        if reg:
            if np.array(check_reg).any() > 0:
                message(
                    "De-regularizing function"
                    " for total RMS deviation"
                    " computation",
                    message_verbosity=2,
                )

                recon_func = SHDeRegularize(
                    recon_func, theta_grid, check_reg, order=reg_order
                )

        Rerr, Amin, Amax = RMSerrs(orig_func, recon_func, info)
        err_info_dict = {"RMS": Rerr, "Amin": Amin, "Amax": Amax}

        result.error_info = err_info_dict
        result.residuals = all_res
        result.residual_axis = np.arange(-1, ell_max + 1)

        if Rerr > 0.1:
            message("Residue warning!", message_verbosity=0)

    return result


def SHContract(modes, info=None, ell_max=None):
    """Reconstruct a function on a grid given its SH modes.

    Parameters
    ----------
    modes : list
            A list of modes, in the convention [[l, [m list]], ]
    info : surfacegridinfo
           An instance of the surfacegridinfo.
    ell_max : int
              The max l mode to include.
    Returns
    -------
    recon_func : ndarray
                 The reconstructed grid function.
    """

    # if isinstance(modes, SingleMode):
    # message("SingleMode obj input. Converting to modes dictionary", message_verbosity=3)

    # modes = modes.get_modes_dict()
    if info is None:
        info = modes.Grid

    if ell_max is None:
        ell_max = modes.ell_max

    # message(f"Modes in SHContract {modes}", message_verbosity=4)

    # print(modes)
    from waveformtools.waveforms import construct_mode_list

    # Construct modes list
    modes_list = construct_mode_list(ell_max=ell_max, spin_weight=0)

    message(f"Modes list in SHContract {modes_list}", message_verbosity=4)

    theta_grid, phi_grid = info.meshgrid

    recon_func = np.zeros(theta_grid.shape, dtype=np.complex128)

    for ell, emm_list in modes_list:
        for emm in emm_list:
            # Clm = modes[f"l{ell}"][f"m{emm}"]

            Clm = modes.mode(ell, emm)
            message(f"Clm shape in SHContract {Clm.shape}", message_verbosity=4)

            recon_func += Clm * Yslm_vec(
                spin_weight=0,
                ell=ell,
                emm=emm,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

    return recon_func

def rotate_polarizations(wf, alpha):
    """Rotate the polarizations of the time domain
    observer waveform by :math:`2\alpha`

    Parameters
    ----------
    wf : 1d array
         The complex observer waveform to rotate.
    alpha : float
            The coordinate angle to rotate the polarizations
            in radians. Note that the polarizarions would
            rotate by :math:`2 \alpha` on a cordinate
            rotation of :math:`\alpha`.

    Returns
    -------
    rot_wf : 1d array
             The rotated waveform.
    """

    h1, h2 = wf.real, wf.imag

    rh1 = np.cos(2 * alpha) * h1 - np.sin(2 * alpha) * h2
    rh2 = np.sin(2 * alpha) * h1 + np.cos(2 * alpha) * h2

    return rh1 + 1j * rh2
