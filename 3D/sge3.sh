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


/share/apps/matlab/R2015b/bin/matlab -r buildDataSet > buildDataSet.txt
/share/apps/matlab/R2015b/bin/matlab -r shuffle > shuffle.txt
julia 3D1.jl > 3D1.txt
#julia 3Dforw.jl > 3Dforw.txt
#/share/apps/matlab/R2015b/bin/matlab -r data > data.txt
#/share/apps/matlab/R2015b/bin/matlab -r svm1 > svm1.txt
#/share/apps/matlab/R2015b/bin/matlab -r svm > svm.txt



