import h5py, numpy as np
import config
import matplotlib
import matplotlib.pyplot as plt

config.conf_matplolib()
import sys

from sxstools.transforms import GetSphereRadialExtents, GetDomainRadii
from config import verbosity

tverb = verbosity.levels()
tverb.set_print_verbosity(2)
from spectral.spherical.grids import GLGrid
import mpi4py
from mpi4py import MPI
from waveformtools.waveformtools import message
from spectral.interpolation.interpolate_new import Interpolate3D
from pathlib import Path

mpi_comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
mpi_nprocs = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()




sim = "q1a0_test_ah2"
run_dir = Path(f"/mnt/pfs/vaishak.p/sims/SpEC/gcc/{sim}/Ev/Lev1_AA/Run")

tsteps = ["Step000001"]

############################################################################################
# Load AH coords (the output grid)
###################################

file = run_dir / "ApparentHorizons/HorizonsDump.h5"

dfileH1 = h5py.File(file)
h1 = dfileH1["AhA.dir"]
hh1 = h1["CoordsMeasurementFrame.tdm"]
h1coords = hh1["data.dat"]
h1coords_index = hh1["index.dat"]
ntime_steps, _ = h1coords_index[...].shape


########################################################################################
# Load AH coord centres
########################

fileh = run_dir / "ApparentHorizons/Horizons.h5"

hdat = h5py.File(fileh)

hdat_aha = hdat["AhA.dir"]

hdat_aha_ccentre = hdat_aha["CoordCenterInertial.dat"]


hc_t = hdat_aha_ccentre[...][:, 0]
hc_x = hdat_aha_ccentre[...][:, 1]
hc_y = hdat_aha_ccentre[...][:, 2]
hc_z = hdat_aha_ccentre[...][:, 3]


##################################################################################
# Load volume four metric
##########################


filev = run_dir / "IHPsiKappa/Vars_SphereA0.h5"
vars_dat = h5py.File(filev)
nr, ntheta, nphi = vars_dat["psi"]["Step000000"].attrs["Extents"]

input_data = []

for item in tsteps:
    trial_func = vars_dat["psi"][item]["tt"][...].reshape(nphi, ntheta, nr).T
    input_data.append(trial_func)

message(f"Raw data shape {np.array(input_data).shape}")
###########################################################################
# Get domain radii
###################

filed = run_dir / "GrDomain.input"
radii = GetDomainRadii(filed)
if mpi_rank == 0:
    message("Radii ", radii)
spa_rad = radii["SphereC"]
r1, r2 = GetSphereRadialExtents(radii, sub_domain="SphereC0")
info = GLGrid(L=ntheta - 1)
theta_grid, phi_grid = info.meshgrid
ntheta, nphi = info.shape


##########################################################
# Construct output grid coords
##############################

# Random coord
if mpi_rank == 0:
    from random import randint

    # index_x = randint(0, ntheta-1)
    # index_y = randint(0, nphi-1)

    index_x = 8
    index_y = 19
    message(f"Chosen indices X{index_x} Y{index_y}")
    message(
        f"Ang Coord values Theta{theta_grid[index_x, index_y]} Phi{phi_grid[index_x, index_y]}"
    )
else:
    index_x = None
    index_y = None

index_x = 8
index_y = 19

r_grid = r1 * np.ones(theta_grid.shape)
one_set_of_coords = np.array(
    [
        [
            r_grid[index_x, index_y],
        ],
        [
            theta_grid[index_x, index_y],
        ],
        [
            phi_grid[index_x, index_y],
        ],
    ]
)

if mpi_rank == 0:
    message(f"Chosen R value {r1}")
    message(f"One set of coords {one_set_of_coords}")


sphp_output_grid_list = np.array(
    [
        one_set_of_coords,
    ]
)


message("SPHP output grid shape ", sphp_output_grid_list.shape)
message(f"SPHP output grid (before Interpolate3D) {sphp_output_grid_list}")

########################################################
# Interpolate
##############

interp3d = Interpolate3D(
    sphp_output_grid=sphp_output_grid_list,
    raw_data=input_data,
    r_min=r1,
    r_max=r2,
    run_dir=run_dir,
)


message("Rank", interp3d.mpi_rank)

if mpi_rank == 0:
    message(
        "SPHP output grid (after Interpolate3D) \t", interp3d.sphp_output_grid
    )
    message("Nprocs \t", interp3d.mpi_nprocs)
    message("Shape of input data \t", interp3d.shape)
    message("Radial grid shape \t", interp3d.radial_grid.Nfuncs)
    message("Ang grid shape \t", interp3d.input_ang_grid.shape)

if mpi_rank == 0:
    message("Interpolating...")

interp3d.construct_interpolant()

if mpi_rank == 0:
    message("Done")
    message("Example mode 2 2 0:")
    message(interp3d.interpolant.mode(2, 2, extra_indices=(0,)))
    message(
        f"Mode shape {interp3d.interpolant.mode(2, 2, extra_indices=(0,)).shape}"
    )
    message("Evaluating...")

    all_zeros = (interp3d.interpolant._modes_data == 0).all()

    message(f"Are all the modes zeros? ... {all_zeros}")

interp3d.evaluate_interpolant()

if mpi_rank == 0:
    message("Done evaluating")
    message("Interpolated data")
    message("\n\t Shape", np.array(interp3d.interpolated_data).shape)
    message("\n\t Data", interp3d.interpolated_data)
    # message("Input data")

    message("Difference")

    for index in range(len(tsteps)):
        message(f"Testing at time step {index}")

        message(f"Input data \n {input_data[index][0][index_x, index_y]}")

        message(
            "Difference",
            np.array(interp3d.interpolated_data[index])
            - np.array([input_data[index][0][index_x, index_y]]),
        )

        np.testing.assert_array_almost_equal(
            interp3d.interpolated_data[index].real,
            [input_data[index][0][index_x, index_y]],
            8,
            "The angular data at the"
            "first radial collocation point"
            "must be recovered to an accuracy"
            "of 12 decimals",
        )
