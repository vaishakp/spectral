import h5py, numpy as np
import config
import matplotlib
import matplotlib.pyplot as plt
import time

config.conf_matplolib()
import sys

sys.path.append("/mnt/pfs/vaishak.p/Projects/Codes/custom_libraries/sxstools")
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
from sxstools.data_loader import SXSDataLoader

mpi_comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
mpi_nprocs = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()


run_dir = (
    "/mnt/pfs/anuj.mishra/spec_runs/test_runs/test_dumptensor_3/Ev/Lev1_AA/Run"
)

# Number of time steps
n_t_steps = 1

##########################################################
# Load grid structure
#####################
dl = SXSDataLoader(
    run_dir=run_dir,
    subdomain="SphereC0",
    metric_dir="PsiKappa",
)

dl.load_grid_structure()


##################################################################################
# Load volume four metric
##########################


input_data = []

for t_step in range(1, n_t_steps + 1):
    input_data.append(dl.get_four_metric(t_step=t_step, component="tt"))

input_data = np.array(input_data)
message(f"Raw data shape {input_data.shape}")


##########################################################
# Construct output grid coords (co-moving)
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
        f"Ang Coord values Theta{dl.theta_grid[index_x, index_y]} Phi{dl.phi_grid[index_x, index_y]}"
    )
else:
    index_x = None
    index_y = None

index_x = 8
index_y = 19

r_grid = dl.r_min * np.ones(dl.AngularGrid.shape)

one_set_of_coords = np.array(
    [
        [
            r_grid[index_x, index_y],
        ],
        [
            dl.theta_grid[index_x, index_y],
        ],
        [
            dl.phi_grid[index_x, index_y],
        ],
    ]
)

if mpi_rank == 0:
    message(f"Chosen R value {dl.r_min}")
    message(f"One set of coords {one_set_of_coords}")


# constructed as a list of (list of) coordinates
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


# Initialize context
interp3d = Interpolate3D(
    sphp_output_grid=sphp_output_grid_list,
    raw_data=input_data,
    r_min=dl.r_min,
    r_max=dl.r_max,
    run_dir=run_dir,
    sxs_data_loader=dl,
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


###################################################
# Construct interpolant


interp3d.construct_interpolant()

if mpi_rank == 0:
    message("Done")
    message("Example mode 2 2 0:")
    message(interp3d.interpolant.mode(2, 2, extra_indices=(0,)))
    message(
        f"Mode shape {interp3d.interpolant.mode(2, 2, extra_indices=(0,)).shape}"
    )
    message("Evaluating...")

    all_zeros = (interp3d.interpolant.modes_data == 0).all()

    message(f"Are all the modes zeros? ... {all_zeros}")


# Evaluate interpolant
start = time.time()
interp3d.evaluate_interpolant()
end = time.time()


print(f"Time taken {end-start}")

#########################################################
# Diagnostics
#############


if mpi_rank == 0:
    message("Done evaluating")
    message("Interpolated data")
    message("\n\t Shape", np.array(interp3d.interpolated_data).shape)
    message("\n\t Data", interp3d.interpolated_data)
    # message("Input data")

    message("Difference")

    for index in range(n_t_steps):
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
            7,
            "The angular data at the"
            "first radial collocation point"
            "must be recovered to an accuracy"
            "of 12 decimals",
        )
