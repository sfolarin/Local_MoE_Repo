module load PrgEnv-gnu cray-mpich/8.1.30
export LMP_MOE=/path/to/lammps_moe
export LD_LIBRARY_PATH="/opt/cray/pe/mpich/8.1.30/ofi/gnu/12.3/lib:$LMP_MOE/build-mpi:$LD_LIBRARY_PATH"
