#!/bin/sh
#$ -N CNN3DSplit1
#$ -S /bin/bash
#$ -q ai.q 
#$ -l h="ahtapot-5-1"
#$ -cwd
#$ -o julia.$JOB_ID.out
#$ -e julia.$JOB_ID.err
#$ -M szia13@ku.edu.tr
#$ -m bea 
export PATH=/share/apps/julia/julia-0.4.0/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/julia/julia-0.4.0/lib:$LD_LIBRARY_PATH


/share/apps/matlab/R2015b/bin/matlab -r buildDataSet > buildDataSet.txt  #builds the dataset
/share/apps/matlab/R2015b/bin/matlab -r shuffle > shuffle.txt #shuffles the dataset for training		
julia 3D1.jl > 3D1.txt #trains the 3D network from scratch
julia 3Dforw.jl > 3Dforw.txt #extracts 3D features from the trained model
/share/apps/matlab/R2015b/bin/matlab -r svm1 > svm1.txt #trains the features using sum



