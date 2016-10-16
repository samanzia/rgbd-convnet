#!/bin/sh
#$ -N VGGextractfeatures
#$ -S /bin/bash
#$ -q ai.q
#$ -l h="ahtapot-5-1"
#$ -cwd
#$ -o julia.$JOB_ID.out
#$ -e julia.$JOB_ID.err


export PATH=/share/apps/julia/julia-0.4.0/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/julia/julia-0.4.0/lib:$LD_LIBRARY_PATH

#copy the features .mat file from both 2D and 3D folder along with their labels
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r combineFeatures > combineFeatures.txt #combines the features
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r svmFusion > svmFusion.txt #trains the features using SVM
