# Implementing a test for the RGB-d dataset
# 148x148 - one layer

using CUDArt
device(2)

using Knet, MAT, ArgParse, CUDNN, JLD



function Knettest(args=ARGS)
	
global softm = JLD.load("bestsoftmax.jld", "model");
w8 = to_host(softm.reg[2].out);
b8 = to_host(softm.reg[4].out);

save("softmaxweights.jld", "w8", w8, "b8", b8)

    
end

!isinteractive() && !isdefined(:load_only) && Knettest(ARGS)
