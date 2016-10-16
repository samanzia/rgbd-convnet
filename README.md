# rgbd-convnet
Masters' thesis implementation

The dataset being used here the UW rgb-d dataset and can be downloaded from here:
http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/

The toolbox used for all of these experiments is Knet which can be downloaded from here for running all the .jl files:
https://github.com/denizyuret/Knet.jl
This links also has instructions for julia installation

The data preprocessing has been done in Matlab and will need the "MAT" package to be read in julia which can be downloaded using the following command in julia:
Pkg.include("MAT")

All the experiments are carried out on the cluster. The data splits for training and testing are the same as given on the datset website.

Other dependancies:

Liblinear

VGG weights (this one uses a modified(but identical) version of http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat found at matconvnet website, to use the model as it is, modify VGGforw.jl accordingly)

Cluster address for modified VGG weights used in 2D experiments:
/mnt/home/szia13/RS/resizedata224/split3

Replication steps:

Run script2D.sh for getting the best performing 2D features

Run script3D.sh for getting the best performing 3D features

Copy the features to the fusion folder

Run scriptFusion.sh to get the best performance
