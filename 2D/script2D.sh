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

#this is for getting the 2D features 
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r buildDataSet > buildDataSet.txt #builds the dataset
julia VGGforw.jl > vggforw.txt #extracts the features using VGG net weights
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r svm > svm.txt #trains the features using SVM
