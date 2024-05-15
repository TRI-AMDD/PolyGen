#!/bin/bash
#$ -pe mpi 8
#$ -cwd
#$ -N PEO_LITFSI
#$ -j y
#$ -o run.log
#$ -m e
#$ -M test
# requesting 300 hrs wall clock time
#$ -l h_rt=300:00:00
module load openmpi/4.1.1
mpirun -np 8 ~/lammps-29-oct-2020/src/lmp_mpi -in equilibration.in > output.txt