import numpy as np
from waveformtools.integrate import TwoDIntegral
from waveformtools.single_mode import SingleMode
from waveformtools.waveformtools import message
from waveformtools.dataIO import construct_mode_list
from spectral.spherical.swsh import Yslm_vec
from spectral.spherical.Yslm_mp import Yslm_mp


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
    error_info=False,
    auto_ell_max=False,
    res_tol_percent=3,
    reg=False,
    reg_order=1,
    label=None,
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
    error_info : bool
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

    if info.grid_type == "GL" and method_info.swsh_routine == "spherepack":
        message("Using SpherePack routine...", message_verbosity=3)

        results = SHExpandSpack(
            func,
            info,
            method_info,
            error_info,
            res_tol_percent,
            reg,
            reg_order=reg_order,
            label=label,
        )

    elif method_info.swsh_routine == "waveformtools":
        message("Using waveformtools routine...", message_verbosity=3)
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
                error_info,
                res_tol_percent,
                reg,
                reg_order=reg_order,
                label=label,
            )

        else:
            message(
                "Using SHExpandSimple:"
                " Expanding upto user prescribed"
                f" ell_max {method_info.ell_max}",
                message_verbosity=3,
            )
            results = SHExpandSimple(
                func,
                info,
                method_info,
                error_info,
                reg=reg,
                reg_order=reg_order,
                label=label,
            )
    elif method_info.swsh_routine == "waveformtools_slow":
        results = SHExpandSimpleSlow(
            func,
            info,
            method_info,
            error_info,
            reg=reg,
            reg_order=reg_order,
            label=label,
        )
    elif method_info.swsh_routine == "spherical":
        results = SHExpandSpherical(
            func,
            info,
            method_info,
            error_info,
            res_tol_percent,
            reg,
            reg_order=reg_order,
            label=label,
        )
    else:
        raise NotImplementedError(
            f"Unknown SWSH method {method_info.swsh_routine}"
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
    error_info=False,
    res_tol_percent=3,
    reg=False,
    reg_order=1,
    check_reg=None,
    label=None,
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
    error_info : bool
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
            message(f"Regularizing function {label}", message_verbosity=2)
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
                raise ValueError(f"Nan found in {label}!")

            Clm = TwoDIntegral(integrand, info, method=method)

            recon_func += Clm * Ylm

            emmCoeffs.update({f"m{emm}": Clm})

        if ell % 2 == 0:
            res2 = np.sqrt(np.mean(np.absolute(func - recon_func) ** 2))

            dres_percent = 100 * (res2 / res1 - 1)

            if dres_percent > res_tol_percent:
                all_res.append(res2)
                message(
                    f" ell_max residue increase error of {dres_percent} % in {label}",
                    message_verbosity=1,
                )

                ell_max = ell - 1
                message(
                    f"Auto setting ell max to {ell_max} instead for {label}",
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

    result = SingleMode(modes_dict=modes, label=label, func=func)
    result._Grid = info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if error_info:
        from waveformtools.diagnostics import RMSerrs

        recon_func = SHContract(modes, info, ell_max)

        ################################
        # Compute total RMS deviation
        # of the expansion
        ###############################

        if reg:
            if np.array(check_reg).any() > 0:
                message(
                    f"De-regularizing function {label} "
                    "for RMS deviation computation",
                    message_verbosity=2,
                )

                recon_func = SHDeRegularize(
                    recon_func, theta_grid, check_reg, order=reg_order
                )

        Rerr, Amin, Amax = RMSerrs(orig_func, recon_func, info)
        error_info_dict = {"RMS": Rerr, "Amin": Amin, "Amax": Amax}

        ############################
        # Update error details
        ############################

        result.error_info = error_info_dict
        result.residuals = all_res

        even_mode_nums = np.arange(0, ell_max, 2)

        residual_axis = [-1] + list(even_mode_nums)

        if ell_max % 2 == 1:
            residual_axis += [ell_max]

        result.residual_axis = residual_axis

        if Rerr > 0.1:
            message(
                f"Residue warning {Rerr} for {label}!  Inaccurate representation.",
                message_verbosity=0,
            )

    #####################################

    return result


def SHExpandSimple(
    func,
    grid_info,
    method_info,
    error_info=False,
    reg=False,
    reg_order=1,
    check_reg=None,
    label=None,
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
    func: ndarray
          The function to be expanded.
    grid_info: Grid
               An instance of the Spherical grid class
               that stores the details of the structure
               of a grid on a topological sphere.
    method_info: MethodInfo
                 An instance of the method info
                 class that contains informations
                 about the numerical methods
                 to be used during the following
                 operations.
    error_info: bool
              Whether or not to compute and return
              the error measures related to the
              SH representation.

    check_reg: list, optional
               A list of two integers (0,1)
               that depicts whether or not to
               regularize the input function
               at the poles.

    Returns
    -------
    modes: dict
           The modes as a dictionary whose keys are lm.

    Notes
    -----
    When regularization is requested,
        1. To compute the total RMS deviation,
           the orginal form is used.
        2. To compute the rms deviation per mode,
           regularized expression is used.


    """
    import sys

    orig_func = func.copy()
    extra_mode_axis_len = len(orig_func.shape) - 2

    theta_grid, phi_grid = grid_info.meshgrid
    ell_max = method_info.ell_max
    int_method = method_info.int_method

    message(
        f"SHExpandSimple: expansion ell max is {ell_max}", message_verbosity=3
    )

    # Check if regularization necessary
    if reg:
        if extra_mode_axis_len > 0:
            raise NotImplementedError(
                f"Regualarizationin {result.label} is not implememted for tensor expansions"
            )

        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message(f"Regularizing function {label}", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    result = SingleMode(ell_max=ell_max, extra_mode_axes_shape=func.shape[:-2])
    result._func = func
    cYslm = Yslm_mp(ell_max=ell_max, spin_weight=0, grid_info=grid_info)
    cYslm.run()

    # integrand = np.conjugate(cYslm.sYlm_modes._modes_data) * func
    integrand = np.einsum(
        "ijk,...jk->i...jk",
        np.conjugate(cYslm.sYlm_modes._modes_data),  # type: ignore
        func,
    )  # type: ignore
    Clm = TwoDIntegral(integrand, grid_info, int_method=int_method)

    result.set_mode_data(Clm)

    result._Grid = grid_info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if error_info:
        result = ComputeErrorInfo(
            result, orig_func, grid_info, ell_max, reg, check_reg, reg_order
        )

    return result


def SHExpandSimpleSlow(
    func,
    grid_info,
    method_info,
    error_info=False,
    reg=False,
    reg_order=1,
    check_reg=None,
    label=None,
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
    error_info : bool
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
    import sys

    orig_func = func.copy()
    theta_grid, phi_grid = grid_info.meshgrid
    ell_max = method_info.ell_max
    int_method = method_info.int_method

    message(
        f"SHExpandSimple: expansion ell max is {ell_max}", message_verbosity=3
    )

    # Check if regularization necessary
    if reg:
        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message(f"Regularizing function {label}", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    result = SingleMode(ell_max=ell_max, label=label, func=func)
    recon_func = np.zeros(func.shape, dtype=np.complex128)

    for ell in range(ell_max + 1):
        emm_list = np.arange(-ell, ell + 1)

        for emm in emm_list:
            Ylm = Yslm_vec(
                spin_weight=0,
                emm=emm,
                ell=ell,
                theta_grid=theta_grid,
                phi_grid=phi_grid,
            )

            integrand = func * np.conjugate(Ylm)
            check_nan(integrand)
            Clm = TwoDIntegral(integrand, grid_info, int_method=int_method)
            recon_func += Clm * Ylm
            # message("Clm ", Clm, message_verbosity=2)
            result.set_mode_data(ell=ell, emm=emm, value=Clm)

    result._Grid = grid_info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if error_info:
        result = ComputeErrorInfo(
            result, orig_func, grid_info, ell_max, reg, check_reg, reg_order
        )

    return result


def check_nan(data):
    """Check for nans in data."""
    uu = np.isnan(data.any())

    if uu:
        raise ValueError("Nan found!")


def ComputeErrorInfo(
    result, orig_func, grid_info, ell_max, reg, check_reg, reg_order
):
    """Compute the RMS errors of an expansion and add to the
    modes obj. This uses wftools optimized SWSH computation
    by default."""

    from waveformtools.diagnostics import RMSerrs
    from spectral.spherical.Yslm_mp import Yslm_mp

    theta_grid, phi_grid = grid_info.meshgrid
    # recon_func = np.zeros(grid_info.shape, dtype=np.complex128)
    # reg_recon_func = np.zeros(grid_info.shape, dtype=np.complex128)
    recon_func = np.zeros(orig_func.shape, dtype=np.complex128)
    reg_recon_func = np.zeros(orig_func.shape, dtype=np.complex128)

    # Compute the full RMS deviation from zero
    Rerr, Amin, Amax = RMSerrs(orig_func, recon_func, grid_info)
    all_res = [Rerr]

    # Compute and cache SHs
    sYlm = Yslm_mp(
        ell_max=ell_max, spin_weight=0, theta=theta_grid, phi=phi_grid
    )
    sYlm.run()

    # Compute unsummed vetor product
    Ylm_vec = sYlm.sYlm_modes._modes_data.transpose((1, 2, 0))
    _, _, modes_data_len = Ylm_vec.shape
    # val_vec = result._modes_data[:modes_data_len] * Ylm_vec
    val_vec = np.einsum(
        "tpm,m...->...tpm", Ylm_vec, result._modes_data[:modes_data_len]
    )

    # Compute powers
    for ell in range(ell_max + 1):
        modes_idx_prev = ell**2
        modes_idx = (ell + 1) ** 2

        reg_recon_func += np.sum(
            val_vec[..., modes_idx_prev:modes_idx], axis=(-1)
        )

        # Deregularize if necessary
        if reg:
            if len(result.extra_mode_axes_shape) > 0:
                raise NotImplementedError(
                    f"Regualarization for {result.label} is not implememted for tensor expansions"
                )

            if np.array(check_reg).any() > 0:
                message(
                    f"De-regularizing function {result.label} "
                    " for total RMS deviation"
                    " computation",
                    message_verbosity=2,
                )
                recon_func = SHDeRegularize(
                    reg_recon_func, theta_grid, check_reg, order=reg_order
                )

        else:
            recon_func = reg_recon_func

        # Compute RMS deviation upto this ell
        Rerr, Amin, Amax = RMSerrs(orig_func, recon_func, grid_info)
        all_res.append(Rerr)

    error_info_dict = {"RMS": Rerr, "dAmin": Amin, "dAmax": Amax}
    result.error_info = error_info_dict
    result.residuals = all_res
    result.residual_axis = np.arange(-1, ell_max + 1)

    # conv = round(100 * Rerr / all_res[0], 2)
    conv = round(100 * np.amax(abs(Rerr)) / np.amax(abs(all_res[0])), 2)

    # if all_res[0] > 1e-8 and conv > 10:
    if np.amax(abs(all_res[0])) > 1e-10 and conv > 10:
        message(
            f"{conv}% Residue warning for {result.label}! ", message_verbosity=0
        )
        message(f"Error report for {result.label}: \n\t {result.error_info}")

    return result


def SHExpandSpack(
    func,
    grid_info,
    method_info,
    error_info,
    res_tol_percent,
    reg,
    reg_order,
    check_reg=None,
    label=None,
):
    """Expand using spherepack.
    Please note that this is only useful for
    expanding real functions !!

    """

    message("Warning: spherepack only works with real functions!")

    from spharm import Spharmt

    nlats = grid_info.L + 1
    nlons = 2 * nlats

    theta_grid, _ = grid_info.meshgrid

    xcls = Spharmt(nlons, nlats, legfunc="computed", gridtype="gaussian")

    # Check if regularization necessary
    if reg:
        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message(f"Regularizing function {label}", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    spack_modes = xcls.grdtospec(func, ntrunc=grid_info.L)

    result = modes_spack_to_wftools(
        spack_modes, func, grid_info, error_info, ell_max=grid_info.L
    )

    if error_info:
        result = ComputeErrorInfo(
            result, func, grid_info, grid_info.L, reg, check_reg, reg_order
        )

    return result


def SHExpandSpherical(
    func,
    grid_info,
    method_info,
    error_info,
    res_tol_percent,
    reg,
    reg_order=1,
    check_reg=None,
    label=None,
):
    """Expand using spherical package."""

    orig_func = func.copy()
    theta_grid, phi_grid = grid_info.meshgrid
    ell_max = method_info.ell_max
    int_method = method_info.int_method

    message(
        f"SHExpandSimpleSpherical: expansion ell max is {ell_max}",
        message_verbosity=3,
    )

    result = SingleMode(ell_max=ell_max, label=label, func=func)

    from spectral.spherical.swsh import create_spherical_Yslm_modes_array

    theta_grid, _ = grid_info.meshgrid

    # Check if regularization necessary
    if reg:
        if check_reg is None:
            check_reg = CheckRegReq(func)

        if np.array(check_reg).any() > 0:
            message(f"Regularizing function {label}", message_verbosity=2)
            func = SHRegularize(func, theta_grid, check_reg, order=reg_order)

    sYlm = create_spherical_Yslm_modes_array(
        theta=theta_grid, phi=phi_grid, ell_max=ell_max, spin_weight=0
    )

    integrand = np.conjugate(sYlm._modes_data) * func

    Clm = TwoDIntegral(integrand, grid_info, int_method=int_method)

    result.set_mode_data(Clm)

    result._Grid = grid_info

    if reg:
        result.reg_order = reg_order
        result.reg_details = check_reg

    else:
        result.reg_order = 0
        result.reg_details = "NA"

    if error_info:
        result = ComputeErrorInfo(
            result, func, grid_info, grid_info.L, reg, check_reg, reg_order
        )

    return result


def modes_spack_to_wftools(
    spack_modes, func, grid_info, error_info, ell_max, label=None
):
    """Convert the modes in spherepack conventaion to modes
    in wftools notation"""

    modes_list = construct_mode_list(ell_max, 0)

    from waveformtools.single_mode import SingleMode

    wf_modes = SingleMode(
        spin_weight=0, ell_max=ell_max, Grid=grid_info, label=label, func=func
    )

    factor = np.sqrt(2 * np.pi)

    for ell, emm_list in modes_list:
        for emm in range(0, ell + 1):
            # for emm in np.array(emm_list):

            spack_ind = get_spack_mode_index(ell, emm, ell_max)
            this_spack_mode_pm = spack_modes[spack_ind]
            this_spack_mode_mm = np.conjugate(this_spack_mode_pm)
            # if emm==0:
            #    factor = np.sqrt(2*np.pi)
            # tmp_mode = this_spack_mode_pm
            # else:
            #    factor = np.sqrt(2*np.pi)
            # tmp_mode = ((-1)**emm) * (this_spack_mode_pm + 1j*this_spack_mode_mm)#/np.sqrt(2)
            tmp_mode = ((-1) ** emm) * (this_spack_mode_pm)

            wf_mode = tmp_mode * factor
            # message(f"l {ell}, m{emm}")
            wf_modes.set_mode_data(ell=ell, emm=emm, value=wf_mode)
            wf_modes.set_mode_data(
                ell=ell, emm=-emm, value=(-1) ** emm * np.conjugate(wf_mode)
            )

    return wf_modes


def get_spack_mode_index(ell, emm, ell_max):
    """Get the mode index in a spherepack mode array
    given a mode

              #
             ##
            ###
           ####
          #####
         ######
        #######
       ########
      #########
     ##########
    ###########
    """

    assert emm >= 0, "Spherepack only save positive m modes"
    assert abs(emm) <= ell, "abs(emm) must be less than or equal to ell"

    num_modes = (ell_max + 1) * (ell_max + 2) / 2
    message("Total num modes", num_modes, message_verbosity=2)

    n_unocc_levs = ell_max + 1 - (emm + 1)
    message("N fully un occupied levels", n_unocc_levs, message_verbosity=2)

    n_occ_sublevel = ell - emm + 1

    n_unocc_sublevel = (ell_max + 1 - emm) - n_occ_sublevel  # (ell - emm) -1

    message("N unocc sublevel", n_unocc_sublevel, message_verbosity=2)
    n_unocc_modes = n_unocc_levs * (n_unocc_levs + 1) / 2 + n_unocc_sublevel

    message("N unocc modes", n_unocc_modes, message_verbosity=2)
    occ_modes = num_modes - n_unocc_modes

    return int(occ_modes) - 1


def SHContract(modes, grid_info, ell_max, method_info, vectorize=False):
    """Reconstruct a function on a grid given its SH modes
    using the specified method"""

    if method_info.swsh_routine == "waveformtools":
        if vectorize:
            result = SHContractWftoolsVec(modes, grid_info, ell_max)
        else:
            result = SHContractWftools(modes, grid_info, ell_max)

    elif method_info.swsh_routine == "spherepack":
        result = SHContractSpack(modes, grid_info)

    else:
        raise KeyError(f"Unknown swsh routine {method_info.swsh_routine}")

    return result


def SHContractSpack(modes, grid_info, ell_max=None):
    """Reconstruct a function on a grid given its SH modes
    using spherepack."""

    if ell_max is not None:
        raise NotImplementedError(
            "Currently, evaluation upto a given ell < ell_max grid is not supported"
        )

    modes_spack = modes.modes_spherepack

    from spharm import Spharmt

    nlats = grid_info.L + 1
    nlons = 2 * nlats

    xcls = Spharmt(nlons, nlats, legfunc="computed", gridtype="gaussian")

    result = xcls.spectogrd(modes_spack)

    return result


def SHContractWftools(modes, grid_info=None, ell_max=None):
    """Reconstruct a function on a grid given its SH modes
    using waveformtools.

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
    if grid_info is None:
        grid_info = modes.Grid

    if ell_max is None:
        ell_max = modes.ell_max

    # message(f"Modes in SHContract {modes}", message_verbosity=4)
    from waveformtools.waveforms import construct_mode_list

    # Construct modes list
    modes_list = construct_mode_list(ell_max=ell_max, spin_weight=0)
    message(f"Modes list in SHContract {modes_list}", message_verbosity=4)
    theta_grid, phi_grid = grid_info.meshgrid

    recon_func = np.zeros(grid_info.shape, dtype=np.complex128)

    for ell in range(ell_max + 1):
        recon_func += SHContractEllWftools(modes, ell, grid_info)

    return recon_func


def SHContractWftoolsVec(modes, grid_info=None, ell_max=None):
    """Reconstruct a function on a grid given its SH modes
    using waveformtools.

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
    if grid_info is None:
        grid_info = modes.Grid

    if ell_max is None:
        ell_max = modes.ell_max

    # message(f"Modes in SHContract {modes}", message_verbosity=4)
    from waveformtools.waveforms import construct_mode_list

    # Construct modes list
    #modes_list = construct_mode_list(ell_max=ell_max, spin_weight=0)
    #message(f"Modes list in SHContract {modes_list}", message_verbosity=4)
    theta_grid, phi_grid = grid_info.meshgrid

    # Compute and cache SHs
    sYlm = Yslm_mp(
        ell_max=ell_max, spin_weight=0, theta=theta_grid, phi=phi_grid
    )
    sYlm.run()

    # Compute unsummed vetor product
    Ylm_vec = sYlm.sYlm_modes._modes_data.transpose((1, 2, 0))
    _, _, modes_data_len = Ylm_vec.shape

    recon_func = np.einsum(
        "tpm,m...->...tp", Ylm_vec, modes._modes_data[:modes_data_len]
    )

    return recon_func


def SHContractEllWftools(modes, ell, grid_info=None):
    """Compute the :math:`\\ell` mode contribution of
    a function on a grid given its SH modes
    using waveformtools.

    Parameters
    ----------
    modes : list
            A list of modes, in the convention [[l, [m list]], ]
    grid_info : surfacegridinfo
           An instance of the surfacegridinfo.
    ell: int
              The :math:`\\ell` mode to include.

    Returns
    -------
    recon_func : ndarray
                 The reconstructed grid function.
    """

    # if isinstance(modes, SingleMode):
    # message("SingleMode obj input. Converting to modes dictionary", message_verbosity=3)

    # modes = modes.get_modes_dict()
    if grid_info is None:
        grid_info = modes.Grid

    if ell is None:
        raise ValueError("Please suply a valid mode number")

    # message(f"Modes in SHContract {modes}", message_verbosity=4)

    # print(modes)

    from waveformtools.waveforms import construct_mode_list

    theta_grid, phi_grid = grid_info.meshgrid

    recon_func_comp = np.zeros(theta_grid.shape, dtype=np.complex128)

    for emm in range(-ell, ell + 1):

        Clm = modes.mode(ell, emm)
        message(f"Clm shape in SHContract {Clm.shape}", message_verbosity=4)

        recon_func_comp += Clm * Yslm_vec(
            spin_weight=0,
            ell=ell,
            emm=emm,
            theta_grid=theta_grid,
            phi_grid=phi_grid,
        )

    return recon_func_comp


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
