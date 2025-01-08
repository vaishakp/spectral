import numpy as np
import math
from math import factorial
from spectral.spherical.swsh import check_Yslm_theta, check_Yslm_args
from waveformtools.waveformtools import message


class Yslm_full_vec:
    """A container to compute the SWSH,
    vectorized over the mode numbers as well
    """

    def __init__(
        self,
        Grid,
        ell_max,
        spin_weight=0,
    ):

        self._spin_weight = spin_weight
        self._Grid = Grid
        self._ell_max = ell_max

        assert (
            spin_weight == 0
        ), "This has only been implemented for spin_weight 0"

    @property
    def Grid(self):
        return self._Grid

    @property
    def ell_max(self):
        return self._ell_max

    @property
    def modes_grid(self):
        return self._modes_grid

    @property
    def spin_weight(self):
        return self._spin_weight

    @property
    def n_ells(self):
        return self._n_ells

    def construct_modes_grid(self):
        """Get grid modes in ell,emm for
        vectorization over modes"""

        ell_grid = np.array((self.ell_max + 1) * [np.arange(self.ell_max + 1)])

        meta_emm_grid = ell_grid.T

        for row_index in range(1, self.ell_max + 1):
            for col_index in range(row_index):

                meta_emm_grid[row_index, col_index] = -row_index + col_index

        self._modes_grid = ell_grid, meta_emm_grid

    def compute_modes(self):

        theta_grid, phi_grid = self.Grid.meshgrid

        ntheta, nphi = theta_grid.shape

        ell_max = ntheta - 1

        assert (
            ell_max >= self.ell_max
        ), "The current grid does not support the requested ell_max"

        theta_grid = check_Yslm_theta(theta_grid)

        from math import comb

        fact = math.factorial

        theta_grid = np.array(theta_grid)
        phi_grid = np.array(phi_grid)

        Sum = np.zeros(
            (self.ell_max + 1, self.ell_max + 1, ntheta, nphi),
            dtype=np.complex256,
        )

        factor = 1

        if self.spin_weight < 0:
            factor = (-1) ** ell
            theta_grid = np.pi - theta_grid
            phi_grid += np.pi

        abs_spin_weight = abs(self.spin_weight)

        for aar in range(0, ell - abs_spin_weight + 1):
            subterm = 0

            if (aar + abs_spin_weight - emm) < 0 or (
                ell - aar - abs_spin_weight
            ) < 0:
                message(f"Skipping r {aar}", message_verbosity=4)
                continue
            else:
                term1 = comb(ell - abs_spin_weight, aar)
                term2 = comb(
                    ell + abs_spin_weight, aar + abs_spin_weight - emm
                )
                term3 = np.power(float(-1), (ell - aar - abs_spin_weight))
                term4 = np.exp(1j * emm * phi_grid)
                term5 = np.longdouble(
                    np.power(
                        np.tan(theta_grid / 2),
                        (-2 * aar - abs_spin_weight + emm),
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

        Yslm_vec_cache[spin_weight][ell_max].update({f"l{ell}m{emm}": value})

        return value
