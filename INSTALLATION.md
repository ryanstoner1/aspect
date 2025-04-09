## Authors
Ryan Stoner <rstoner@miami.edu>, Rene Gassmoeller, @gassmoeller

Portions of the text and formatting below were taken directly from the instructions for [Installation on TACC Stampede](Compiling-and-Running-ASPECT-on-TACC-Stampede) written by Sarah Stamps, Jonathan Perry-Houts and Eric Heien, [Installation on TACC Stampede2](Compiling-and-Running-ASPECT-on-TACC-Stampede2) written by John Naliboff, Haoyuan Li, and Juliane Dannberg, and [Installation on TACC Frontera](https://github.com/geodynamics/aspect/wiki/Installation-on-Frontera) by Timo  Heister and Rene Gassmoeller.


## Before you begin:

- to connect: ``ssh your-tacc-user-id@stampede3.tacc.utexas.edu``
- do not work/compile on the login node, but submit an interactive job: ``idev -m 120``
- use $SCRATCH or $WORK to compile and install, not $HOME (edit: earlier only $SCRATCH worked)
- running binaries with MPI does not work (hang) unless you use ``ibrun``, for example``ibrun -n X ./aspect``, leaving out the ``-n X`` will use however many cores are available within this job


## Overview
Please note that many parts of the documentation have changed from the instructions on the original Stampede and Stampede2. Even if you have set up ASPECT before, please follow all sections of this document to make sure your current setup is installed and configured correctly.

Also note that the instructions below so far refer strictly to installation on the Stampede3 SKX and ICX nodes. If you have experience with the SPR nodes, please update this guide.

## Account 

You first need to obtain an ACCESS account at https://allocations.access-ci.org/. If you need one send an email to: cig-help@geodynamics.org. 

When you have an account created on ACCESS and are assigned resources on Stampede3, a TACC (Texas Advanced Computing Center) account will automatically be created with information sent to the email address associated with your ACCESS account. If you already have a TACC account, additional steps may be needed to link your ACCESS and TACC accounts. Notably, you will also need to setup the TACC two-factor authentication app (TACC Token) on a mobile device if you wish to login into to login into Stampede3 directly via ssh: _ssh tacc_username@stampede3.tacc.utexas.edu_.

## Installation

### For deal.II 

Many of the packages required by ASPECT (Trilinos, p4est, HDF5, NetCDF) can be activated as modules. However, deal.II is also required. To speed up the installation significantly we will also use a development node, which provides access to 48 cores.

1. After logging into Stampede3, start by loading the required modules and set up environment variables.

Add the following lines to the file `.bashrc` in your home directory:
```
module purge
module load intel/24.0 impi/21.11 autotools/1.3 xalt/3.0. TACC netcdf/4.9.2 p4est/2.8.5 trilinos/15.0.0 phdf5/1.14.3 boost/1.85.0
```

Now log out and log in again to Stampede. If you run the command `module list` you should see the following output:

```
Currently Loaded Modules (order may vary):
  1) intel/24.0   3) autotools/1.4   5) xalt/3.1.1   7) hdf5/1.14.4    9) phdf5/1.14.3  11) trilinos/15.0.0
  2) impi/21.11   4) cmake/3.31.5    6) TACC         8) netcdf/4.9.2  10) p4est/2.8.5   12) boost/1.85.0   

```

2. Next we will compile the remaining dependencies for ASPECT including deal.II. We start by going to the "$WORK" directory and making a series of folders:

```
cd $WORK 
mkdir software 
cd software 
```
While the directory structure and folder names above can be modified, this will require changing the specified PATHS in a number of steps below. Next we ask for an interactive development node on Stampede3 to compile:

```
idev -p skx-dev -N 1 -n 48 -m 120
```
This command will (possibly after a little wait) open a new terminal on an interactive development node. Respectively, ```N, n and m``` refer to the number of nodes, cores and minutes for the interactive session. Note that your account will be charged based on the length of the session, rather than the requested total number of minutes. If you have access to more than one ACCESS allocation with time on Stampede3, you will need to specify the account number by adding _-A account_number_ to the idev command.

3. Download and configure deal.II in the directory using `candi`:

Below we install version 9.5.2 of deal.II. If you prefer a development version set `STABLE_BUILD=true` and `DEAL_II_VERSION=master`, although there may be incompatibilities between future deal.II versions and the available compilers.

```
cd $WORK2/software
git clone https://github.com/dealii/candi.git
cd candi
```

Next you need to create a file called `local.cfg` inside the candi directory. This file should contain the following text (copy and paste to avoid typos):
```
NATIVE_OPTIMIZATIONS=OFF
BUILD_EXAMPLES=OFF
USE_DEAL_II_CMAKE_MPI_COMPILER=OFF

DEAL_II_CONFOPTS="-D DEAL_II_COMPILER_HAS_RESTRICT_KEYWORD=no -D DEAL_II_WITH_COMPLEX_VALUES=OFF -D DEAL_II_COMPONENT_EXAMPLES=OFF -D DEAL_II_WITH_64BIT_INDICES=ON -D DEAL_II_CXX_FLAGS='-Wno-tautological-constant-compare -fno-finite-math-only -Wno-deprecated-declarations -march=native'"

PACKAGES="load:dealii-prepare once:astyle once:sundials dealii"

MKL=ON
MKL_DIR=$TACC_MKL_LIB
P4EST_DIR=$TACC_P4EST_DIR
TRILINOS_DIR=$TACC_TRILINOS_DIR
HDF5_DIR=$TACC_HDF5_DIR
NETCDF_DIR=$TACC_NETCDF_DIR

DEAL_II_VERSION=v9.5.2
```

Now we can execute candi to install astyle, sundials and deal.II, we need to specify the platform as centos7 since candi does not recognize the Stampede3 operating system. We also have to add the configuration to your environment `$HOME/.bashrc` so that ASPECT can find deal.II automatically:

```
./candi.sh --platform=deal.II-toolchain/platforms/supported/centos7.platform -j48 -p $WORK/software/deal.II-v9.5.2
```

After this step is completed candi will print a path that we can `source`. Add it to `.bashrc`:

```
echo ". $WORK/software/deal.II-v9.5.2/configuration/enable.sh" >> $HOME/.bashrc
source $HOME/.bashrc
```

4. Fix include directory MPI version (for ASPECT versions 2.5.0, ASPECT 2.6.0 does not need this step).

The latest intel MPI information isn't correctly included in deal.II. We need to manually edit this information in _config.h_ for ASPECT to compile.

```
cd $WORK/software/deal.II-master/deal.II-master/include/deal.II/base
```
Replace the lines (450-451)
 
```
#  define DEAL_II_MPI_VERSION_MAJOR 
#  define DEAL_II_MPI_VERSION_MINOR 
```
with
```
#  define DEAL_II_MPI_VERSION_MAJOR 3
#  define DEAL_II_MPI_VERSION_MINOR 1
```

5. Download, configure and install ASPECT version

Download:
```
cd $WORK2/software 
git clone https://github.com/geodynamics/aspect.git
cd aspect
```
Switch to the 2.5 release branch (if using ASPECT 2.5):
```
git checkout -b aspect-2.5 origin/aspect-2.5
```

Configure and install ASPECT:
```
mkdir build
cd build
cmake -D ASPECT_ADDITIONAL_CXX_FLAGS="-fno-finite-math-only" -D CMAKE_CXX_COMPILER=icpx ..
make -j48
```
## Submitting jobs via sbatch

Any runs that require more than 4 nodes or multi-hour run times will require submitting jobs to a "normal" compute node (i.e., not development) to the Stampede3 job scheduler system (SLURM). This is accomplished by submitting a job script via the `sbatch` command. 

An example job script is posted below, which would be submitted with `sbatch file_name.sh` (e.g., if the file is named `run.sh` then the command is `sbatch run.sh`). Significantly, note the selected partition (i.e., node type) is skx-normal. Given that we compiled on an skx node above via the interactive idev session, ASPECT and all of the dependencies will only be able to run optimally on the skx nodes (skx-normal or skx-dev). Newer nodes (icx or later) may work as well (untested at this time).


```
#!/bin/bash
#SBATCH -J aspmod1
#SBATCH -o log_aspect_model_1
#SBATCH -e %j
#SBATCH -p skx
#SBATCH -t 24:00:00
#SBATCH -N 4
#SBATCH -n 192

# Aspect executable
ASP="$WORK2/software/aspect/build/aspect-release"

# Run model.  Submit job with "sbatch run.sh"
ibrun $ASP aspect_model_1.prm
```
