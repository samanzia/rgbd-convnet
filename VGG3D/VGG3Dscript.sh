#!/bin/sh
#$ -N VGGextractfeatures
#$ -S /bin/bash
#$ -q ai.q
#$ -l h="ahtapot-5-1"
#$ -cwd
#$ -o julia.$JOB_ID.out
#$ -e julia.$JOB_ID.err

#ensure to include vgg-verydeep-16.mat in working directory
export PATH=/share/apps/julia/julia-0.4.0/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/julia/julia-0.4.0/lib:$LD_LIBRARY_PATH

#this creates a .mat files for each file in the dataset by preprocessing the depth and creating a 3D voxel with 6 channels
#all details are in the thesis
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r voxelize > voxelize.txt 
#this builds the dataset from the preprocessed files
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r buildDataSet > buildDataSet.txt
#this reshapes the 3D dataset to 2D so that they can input to VGG net
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r reshape3Dto2D > reshape3Dto2D.txt
#this extracts features from the dataset so that the softmax layer can be trained
julia VGGforw.jl > VGGforw.jl
#this shuffles the extracted features for training
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r shuffleFeatures > shuffleFeatures.txt
#this trains the softmax layer to be incorporated in VGG3D
julia trainSoftmaxLayer.jl > trainSoftmaxLayer.txt
#this extracts the softmax weights from the best model
julia extractSoftmaxWeights.jl > extractSoftmaxWeights.txt
#this shuffles the 3D dataset for training on VGG3D
/share/apps/matlab/R2015b/bin/matlab -nodisplay -r shuffle3D > shuffle3D.txt
#this trains the VGG3D network and gives the final performance. The saved model can also be used to extract features for fusion
julia VGG3Dtrain.jl > VGG3Dtrain.txt

