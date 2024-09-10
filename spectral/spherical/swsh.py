import math

import numpy as np
import sympy as sp
from waveformtools.waveformtools import message


def check_Yslm_args(spin_weight, ell, emm):
    """Check if the arguments to a Yslm functions
    makes sense

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    """

    assert ell >= abs(spin_weight), (
        " ell should be greater than"
        "or equal to the absolute value of spin weight "
    )

    assert abs(emm) <= ell, (
        "absolute value of emm should be" "less than or equal to ell"
    )


def Yslm(spin_weight, ell, emm, theta, phi):
    """Spin-weighted spherical harmonics fast evaluation.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic.
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta : float
            The polar angle  :math:`\\theta` in radians,
    phi : float
          The aximuthal angle :math:`\\phi' in radians.

    Returns
    --------
    Yslm : float
           The value of Yslm at :math:`\\theta, phi'.

    Note
    ----
    This is accurate upto 14 decimals for L upto 25.
    """

    check_Yslm_args(spin_weight, ell, emm)
    import sympy as sp

    # theta, phi = sp.symbols('theta phi')

    fact = math.factorial
    # fact = sp.factorial
    Sum = 0

    factor = 1
    if spin_weight < 0:
        factor = (-1) ** ell
        theta = np.pi - theta
        phi += np.pi

    abs_spin_weight = abs(spin_weight)

    for aar in range(ell - abs_spin_weight + 1):
        if (aar + abs_spin_weight - emm) < 0 or (
            ell - aar - abs_spin_weight
        ) < 0:
            message(f"Skippin r {aar}", message_verbosity=3)
            continue
        else:
            Sum += (
                sp.binomial(ell - abs_spin_weight, aar)
                * sp.binomial(
                    ell + abs_spin_weight, aar + abs_spin_weight - emm
                )
                * np.power((-1), (ell - aar - abs_spin_weight))
                * np.exp(1j * emm * phi)
                / np.power(np.tan(theta / 2), (2 * aar + abs_spin_weight - emm))
            )

    Sum = complex(Sum)
    Yslm = (-1) ** emm * (
        np.sqrt(
            fact(ell + emm)
            * fact(ell - emm)
            * (2 * ell + 1)
            / (
                4
                * np.pi
                * fact(ell + abs_spin_weight)
                * fact(ell - abs_spin_weight)
            )
        )
        * np.sin(theta / 2) ** (2 * ell)
        * Sum
    )

    return factor * Yslm


def check_Yslm_theta(theta_grid, threshold=1e-6):
    theta_list = np.array(theta_grid).flatten()

    locs = np.where(abs(theta_list) < threshold)

    for index in locs:
        sign = theta_list[index] / abs(theta_list[index])

        theta_list[index] = theta_list[index] + sign * threshold

    return theta_list.reshape(np.array(theta_grid).shape)

Yslm_vec_cache = {}
def Yslm_vec(spin_weight, ell, emm, theta_grid, phi_grid, cache=True):
    """Spin-weighted spherical harmonics fast evaluations
    on numpy arrays for vectorized evaluations.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta : float
            The polar angle  :math:`\\theta` in radians,
    phi : float
          The aximuthal angle :math:`\\phi' in radians.

    Returns
    --------
    Yslm : float
           The value of Yslm at :math:`\\theta, phi'.

    Note
    ----
    This is accurate upto 14 decimals for L upto 25.
    """

    if cache:
        ntheta, nphi = theta_grid.shape

        ell_max = ntheta-1

        check_Yslm_args(spin_weight, ell, emm)

        flag = query_Yslm_vec_cache(spin_weight=spin_weight, ell_max=ell_max, ell=ell, emm=emm)

    else:
        flag=False

    if flag:
        return Yslm_vec_cache[spin_weight][ell_max][f'l{ell}m{emm}']
    
    else:
        theta_grid = check_Yslm_theta(theta_grid)

        from math import comb

        fact = math.factorial

        theta_grid = np.array(theta_grid)
        phi_grid = np.array(phi_grid)

        Sum = 0 + 1j * 0

        factor = 1
        if spin_weight < 0:
            factor = (-1) ** ell
            theta_grid = np.pi - theta_grid
            phi_grid += np.pi

        abs_spin_weight = abs(spin_weight)

        for aar in range(0, ell - abs_spin_weight + 1):
            subterm = 0

            if (aar + abs_spin_weight - emm) < 0 or (
                ell - aar - abs_spin_weight
            ) < 0:
                message(f"Skipping r {aar}", message_verbosity=4)
                continue
            else:
                term1 = comb(ell - abs_spin_weight, aar)
                term2 = comb(ell + abs_spin_weight, aar + abs_spin_weight - emm)
                term3 = np.power(float(-1), (ell - aar - abs_spin_weight))
                term4 = np.exp(1j * emm * phi_grid)
                term5 = np.longdouble(
                    np.power(
                        np.tan(theta_grid / 2), (-2 * aar - abs_spin_weight + emm)
                    )
                )
                subterm = term1 * term2 * term3 * term4 * term5

                Sum += subterm

        Yslmv = float(-1) ** emm * (
            np.sqrt(
                np.longdouble(fact(ell + emm))
                * np.longdouble(fact(ell - emm))
                * (2 * ell + 1)
                / (
                    4
                    * np.pi
                    * np.longdouble(fact(ell + abs_spin_weight))
                    * np.longdouble(fact(ell - abs_spin_weight))
                )
            )
            * np.sin(theta_grid / 2) ** (2 * ell)
            * Sum
        )

        value = factor * Yslmv

        if np.isnan(np.array(value)).any():
            message(
                "Nan discovered. Falling back to Yslm_prec on defaulted locations",
                message_verbosity=1,
            )

            nan_locs = np.where(np.isnan(np.array(value).flatten()))[0]

            message("Nan locations", nan_locs, message_verbosity=1)

            theta_list = np.array(theta_grid).flatten()
            phi_list = np.array(phi_grid).flatten()

            message("Theta values", theta_list[nan_locs], message_verbosity=1)

            value_list = np.array(value, dtype=np.complex128).flatten()

            for index in nan_locs:
                replaced_value = Yslm_prec(
                    spin_weight=spin_weight,
                    theta=theta_list[index],
                    phi=phi_list[index],
                    ell=ell,
                    emm=emm,
                )

                value_list[index] = replaced_value

            value = np.array(value_list).reshape(theta_grid.shape)

            message("nan corrected", value, message_verbosity=1)

            if np.isnan(np.array(value)).any():
                message(
                    "Nan re discovered. Falling back to Yslm_prec_grid",
                    message_verbosity=1,
                )

                value = np.complex128(
                    Yslm_prec_grid(
                        spin_weight, ell, emm, theta_grid, phi_grid, prec=16
                    )
                )

                if np.isnan(np.array(value)).any():
                    if (abs(np.array(theta_grid)) < 1e-14).any():
                        # print("!!! Warning: setting to zero manually. Please check again !!!")
                        # value = 0
                        raise ValueError(
                            f"Possible zero value encountered due to small theta {np.amin(theta_grid)}"
                        )

                    else:
                        raise ValueError(
                            "Although theta>1e-14, couldnt compute Yslm. Please check theta"
                        )

        if cache:
            Yslm_vec_cache[spin_weight][ell_max].update({f"l{ell}m{emm}" : value})

        return value

            
def query_Yslm_vec_cache(spin_weight, ell_max, ell, emm):
                    
    flag = False
                    
    if spin_weight not in Yslm_vec_cache.keys():
        Yslm_vec_cache.update({spin_weight : {}})
    
    if ell_max not in Yslm_vec_cache[spin_weight].keys():
        Yslm_vec_cache[spin_weight].update({ell_max : {}})
    
    if f'l{ell}m{emm}' not in Yslm_vec_cache[spin_weight][ell_max].keys():
        Yslm_vec_cache[spin_weight][ell_max].update({f"l{ell}m{emm}" : None})

        return False
      
    else:
        return True

def Yslm_prec_grid(spin_weight, ell, emm, theta_grid, phi_grid, prec=24):
    """Spin-weighted spherical harmonics function with precise computations
    on an angular grid. Uses a symbolic method evaluated at the degree
    of precision requested by the user.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta_grid : 2darray
                 The polar angle  :math:`\\theta` in radians,
    phi_grid : 2darray
               The aximuthal angle :math:`\\phi' in radians.
    pres : int, optional
           The precision i.e. number of digits to compute
           upto. Default value is 16.

    Returns
    --------
    Yslm_vals : float
               The value of Yslm at the grid
               :math:`\\theta, phi'.
    """

    theta_grid_1d, phi_grid_1d = theta_grid.flatten(), phi_grid.flatten()
    from itertools import zip_longest

    ang_set = zip_longest(theta_grid_1d, phi_grid_1d)

    Yslm_vals = np.array(
        [
            Yslm_prec(
                spin_weight=spin_weight,
                theta=thetav,
                phi=phiv,
                ell=ell,
                emm=emm,
                prec=prec,
            )
            for thetav, phiv in ang_set
        ]
    ).reshape(theta_grid.shape)

    return Yslm_vals


def Yslm_prec(spin_weight, ell, emm, theta, phi, prec=24):
    """Spin-weighted spherical harmonics function with precise computations.
    Uses a symbolic method evaluated at the degree of precision requested
    by the user.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta : float
            The polar angle  :math:`\\theta` in radians,
    phi : float
          The aximuthal angle :math:`\\phi' in radians.
    pres : int, optional
           The precision i.e. number of digits to compute
           upto. Default value is 16.

    Returns
    --------
    Yslm : float
           The value of Yslm at :math:`\\theta, phi'.
    """

    check_Yslm_args(spin_weight, ell, emm)

    import sympy as sp

    # tv, pv = theta, phi
    th, ph = sp.symbols("theta phi")

    Yslm_expr = Yslm_prec_sym(spin_weight, ell, emm)

    if spin_weight < 0:
        theta = np.pi - theta
        phi = np.pi + phi

    return Yslm_expr.evalf(
        prec, subs={th: sp.Float(f"{theta}"), ph: sp.Float(f"{phi}")}
    )


def Yslm_prec_sym(spin_weight, ell, emm):
    """Spin-weighted spherical harmonics precise,
    symbolic computation for deferred evaluations.
    Is dependent on variables th: theta and ph:phi.

    Parameters
    ----------
    spin_weight : int
                  The Spin weight of the harmonic
    ell : int
          The mode number :math:`\\ell'.
    emm : int
          The azimuthal mode number :math:`m'.
    theta : float
            The polar angle  :math:`\\theta` in radians,
    phi : float
          The aximuthal angle :math:`\\phi' in radians.
    pres : int, optional
           The precision i.e. number of digits to compute
           upto. Default value is 16.

    Returns
    --------
    Yslm : sym
           The value of Yslm at :math:`\\theta, phi'.
    """

    check_Yslm_args(spin_weight, ell, emm)

    import sympy as sp

    th, ph = sp.symbols("theta phi")

    fact = sp.factorial
    Sum = 0

    abs_spin_weight = abs(spin_weight)
    # To get negative spin weight SWSH
    # in terms of positive spin weight
    factor = 1
    if spin_weight < 0:
        factor = sp.Pow(-1, ell)

    for aar in range(ell - abs_spin_weight + 1):
        if (aar + abs_spin_weight - emm) < 0 or (
            ell - aar - abs_spin_weight
        ) < 0:
            # message('Continuing')
            continue
        else:
            # message('r, l, s, m', r, l, s, m)
            # a1 = sp.binomial(ell - spin_weight, aar)
            # message(a1)
            # a2 = sp.binomial(ell + spin_weight, aar + spin_weight - emm)
            # message(a2)
            # a3 = sp.exp(1j * emm * phi)
            # message(a3)
            # a4 = sp.tan(theta / 2)
            # message(a4)

            Sum += (
                sp.binomial(ell - abs_spin_weight, aar)
                * sp.binomial(
                    ell + abs_spin_weight, aar + abs_spin_weight - emm
                )
                * sp.Pow((-1), (ell - aar - abs_spin_weight))
                * sp.exp(sp.I * emm * ph)
                * sp.Pow(sp.cot(th / 2), (2 * aar + abs_spin_weight - emm))
            )

    Yslm_expr = sp.Pow(-1, emm) * (
        sp.sqrt(
            fact(ell + emm)
            * fact(ell - emm)
            * (2 * ell + 1)
            / (
                4
                * sp.pi
                * fact(ell + abs_spin_weight)
                * fact(ell - abs_spin_weight)
            )
        )
        * sp.Pow(sp.sin(th / 2), (2 * ell))
        * Sum
    )

    Yslm_expr = factor * sp.simplify(Yslm_expr)

    return Yslm_expr


def realYlm(theta_grid, phi_grid, ell, emm, method='vec', prec=None, grid_info=None, ell_max=None, nprocs=4):
    ''' The real spherical harmonics
    
    Parameters 
    ----------
    theta_grid,phi_grid : 2darray
                          The meshgrid of angles
    ell, emm : int, >0
               The mode numbers
    method : str
             'vec' for vectorized Yslm computation.
             'prec_grid' for precise Yslm computation. 
             'prec_grid_mp' for precise, parallel computation
             of Yslm. Also specify the precision alongwith for 
             precise computations.
    
    Returns
    -------
    realYlm : 2darray
              The real spin weight 0 spherical harmonics as defined on wikipedia.


    '''
    from spectral.spherical.Yslm_prec_grid_mp import Yslm_prec_grid_mp
    
    spin_weight = 0

    if method == 'vec':
        Yl_pm = Yslm_vec(theta_grid=theta_grid, phi_grid=phi_grid, spin_weight=spin_weight, ell=ell, emm=emm)
        Yl_mm = Yslm_vec(theta_grid=theta_grid, phi_grid=phi_grid, spin_weight=spin_weight, ell=ell, emm=-emm)

    elif method == 'prec_grid_mp':
        

        Ylm_calc = Yslm_prec_grid_mp(ell_max=ell_max, grid_info=grid_info, nprocs=None, spin_weight=0, prec=prec)

        Ylm_calc.run()

        res_ind_p = ell**2 + emm + ell
        res_ind_m = ell**2 -emm + ell


        Yl_pm = Ylm_calc.result_list[res_ind_p][1]
        Yl_mm = Ylm_calc.result_list[res_ind_m][1]


    elif method=='prec_grid':
        Yl_pm = Yslm_prec_grid(theta_grid=theta_grid, phi_grid=phi_grid, spin_weight=spin_weight, ell=ell, emm=emm)
        Yl_mm = Yslm_prec_grid(theta_grid=theta_grid, phi_grid=phi_grid, spin_weight=spin_weight, ell=ell, emm=-emm)

    else:
        raise KeyError(f"The requested method {method} not found. The available methods are:\n"
                        "\t 'vec' for vectorized Yslm computation.\n"
                        "\t 'prec_grid' for precise Yslm computation. \n"
                        "\t 'prec_grid_mp' for precise, parallel computation \n"
                        "\t of Yslm. Also specify the precision alongwith for \n"
                        "\t precise computations.")
    
    if emm<0:
        reYlm = (1j/np.sqrt(2))* (Yl_pm - (-1)**(emm) * Yl_mm)
    elif emm==0:
        reYlm = Yl_pm
    
    else:
        reYlm = (1/np.sqrt(2))* (Yl_mm + (-1)**(emm) * Yl_pm)


    return reYlm


def create_Yslm_modes_array(theta, phi, spin_weight, ell_max):
    ''' Create a modes array containing the sYlm modes at the given 
    values theta and phi 

    Parameters
    ----------    
    theta, phi: float
                The angular points to evaluate at.

    spin_weight: int
                 The spin-weight of the SWSH

    ell_max: int
             The mode number upto which to evaluate

 
    Returns
    -------
    sYlm_modes: SingleMode
                A modes array contining the requested modes.

    '''
    from waveformtools.single_mode import SingleMode
    from waveformtools.dataIO import construct_mode_list

    modes_list = construct_mode_list(spin_weight=spin_weight, ell_max=ell_max)

    sYlm_modes = SingleMode(ell_max=ell_max, 
                            spin_weight=spin_weight,
                            )

    for ell, emm_list in modes_list:
        for emm in emm_list:
            Ylm = Yslm_vec(spin_weight=spin_weight, 
                           theta_grid=theta, 
                           phi_grid=phi, 
                           ell=ell, 
                           emm=emm, 
                           cache=False)
                    
            sYlm_modes.set_mode_data(ell, emm, Ylm)


    return sYlm_modes


def get_spherical_Yslm_index(ell, emm):
    ind = 0
    for ell_ind in range(ell + 1):
        ind += 2 * ell_ind + 1

    return ind + emm - ell_ind - 1
    
    
def create_spherical_Yslm_modes_array(theta, phi, spin_weight, ell_max):
    ''' Create a modes array containing the sYlm modes at the given 
    values theta and phi 

    Parameters
    ----------    
    theta, phi: float
                The angular points to evaluate at.

    spin_weight: int
                 The spin-weight of the SWSH

    ell_max: int
             The mode number upto which to evaluate

 
    Returns
    -------
    sYlm_modes: SingleMode
                A modes array contining the requested modes.

    '''
    from waveformtools.single_mode import SingleMode
    from waveformtools.dataIO import construct_mode_list
    import quaternionic, spherical
    
    modes_list = construct_mode_list(spin_weight=spin_weight, ell_max=ell_max)

    if isinstance(theta, float):
        sYlm_modes = SingleMode(ell_max=ell_max, 
                                spin_weight=spin_weight,
                                )
        
        message("Created a modes array of mode axis length 0", message_verbosity=2)

    elif isinstance(theta, np.ndarray):
        sYlm_modes = SingleMode(ell_max=ell_max, 
                                spin_weight=spin_weight,
                                extra_mode_axis_shape=theta.shape
                                )
        
        message(f"Created a modes array of mode axis shape {theta.shape}", message_verbosity=2)

    #message(f"Created modes axis shape is {sYlm_modes._modes_data.shape}")

    R = quaternionic.array.from_spherical_coordinates(theta, phi)
    wigner = spherical.Wigner(ell_max)
    Y2 = wigner.sYlm(spin_weight, R)

    #message(f"Y2 shape {Y2.shape}")

    for ell, emm_list in modes_list:
        for emm in emm_list:
            #print(type(ell), ell, type(emm), emm)
            ind = get_spherical_Yslm_index(ell, emm)

            #message(f"Y2 ind shape {Y2[ind].shape}")
            
            sYlm_modes.set_mode_data(ell=ell, emm=emm, value=Y2.T[ind].T)

    return sYlm_modes