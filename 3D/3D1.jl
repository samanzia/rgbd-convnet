# for 3-D

using CUDArt
device(0)

using Knet, MAT, ArgParse, CUDNN, JLD

#conv function
@knet function wconv3d(x; outt=0, window=0, cinit=Xavier(), p = 0, o...)
    w = par(; o..., init=cinit, dims=(window, window, window, 0, outt))
    return conv(w,x; padding = p)
end

#bias function
@knet function bias5(x; binit=Constant(0), o...)
    b = par(; o..., init=binit, dims=(1,1,1,0,1))
    return b+x
end

#function for conv+bias+rectification+pooling
@knet function cbfp3d(x; f=:relu, cwindow=0, pwindow=0, pad = 0,out =0,o...)
    y = wconv3d(x; o..., window=cwindow,p = pad, outt=out)
    z = bias5(y; o...)
    r = f(z; o...)
    return pool(r; o..., window=pwindow)
end

#3-D model (weights initialized from scratch)
@knet function threeD_model(x0; h = 0, odim = 0, cdim = 0, pdim = 0, cinit=Gaussian(0.0, 0.01))
    x2 = cbfp3d(x0; out=64, f=:relu, cwindow=3, pwindow=2, pad = 1, cinit=Gaussian(0.0, 0.01))
	x3 = cbfp3d(x2; out=128, f=:relu, cwindow=3, pwindow =2,pad = 1, cinit=Gaussian(0.0, 0.01))
	x4 = wbf(x3; out=4096, f=:relu,winit=Gaussian(0.0, 0.01))
    return wbf(x4; out=51, f=:soft, winit=Gaussian(0.0,0.01))
end

function Knettest(args=ARGS)
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--nbatch"; arg_type=Int; default=100)
        ("--lr"; arg_type=Float64; default= 0.1)
        ("--epochs"; arg_type=Int; default=10)
		
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    #println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    
    #loading dataset (matlab files) using the mat package
	xtrnFile = matopen("TrainShuffleSplit1.mat")
	xtrn = read(xtrnFile,"TrainSplit1")
    xtrn = map(Float32,xtrn)
	println(size(xtrn))

	xtstFile = matopen("TestShuffleSplit1.mat")
	xtst = read(xtstFile,"TestSplit1")
    xtst = map(Float32,xtst)
	println(size(xtst))
	
	ytrnFile = matopen("TrainShuffleSplit1LabelsMat.mat")
	ytrn = read(ytrnFile,"TrainSplit1LabelsMat")
    ytrn = map(Float32,ytrn)
	println(size(ytrn))

	ytstFile = matopen("TestShuffleSplit1LabelsMat.mat")
	ytst = read(ytstFile,"TestSplit1LabelsMat")
	ytst = map(Float32,ytst)
	println(size(ytst))
	
    #normalizing values to lie between 0 and 255
	mtrn = maximum(xtrn);
	xtrn = xtrn/mtrn;
	xtst = xtst/mtrn;
	
    #minibatching
    dtrn = minibatch(xtrn, ytrn, nbatch)
    dtst = minibatch(xtst, ytst, nbatch)
	
    #compiling and training
    threeD = compile(:threeD_model)
    setp(threeD; lr=lr)
	l=zeros(2); m=zeros(2)
    @time for epoch=1:epochs
        train(threeD,dtrn,softloss;losscnt=fill!(l,0), maxnorm=fill!(m,0))
        atrn = 1-test(threeD,dtrn,zeroone)
        atst = 1-test(threeD,dtst,zeroone)
        println((epoch, atrn, atst))
	end
	
    #save model for later use
	JLD.save("threeD.jld", "model", clean(threeD));
	return (l[1]/l[2],m...)
	
end

#helper functions
function train(f, data, loss; losscnt=nothing, maxnorm=nothing)
    for (x,ygold) in data
        ypred = forw(f, x)
        back(f, ygold, loss)
        update!(f)
        losscnt[1] += loss(ypred, ygold); losscnt[2] += 1
        w=wnorm(f); w > maxnorm[1] && (maxnorm[1]=w)
        g=gnorm(f); g > maxnorm[2] && (maxnorm[2]=g)
    end
end

function test(f, data, loss)
    sumloss = numloss = 0
    for (x,ygold) in data
        ypred = forw(f, x)
        sumloss += loss(ypred, ygold)
        numloss += 1
    end
    sumloss / numloss
end

function minibatch(x, y, batchsize)
    data = Any[]
    for i=1:batchsize:ccount(x)-batchsize+1
        j=i+batchsize-1
        push!(data, (cget(x,i:j), cget(y,i:j)))
    end
    return data
end

function getgrad(f, data, loss)
    (x,ygold) = first(data)
    ypred = forw(f, x)
    back(f, ygold, loss)
    loss(ypred, ygold)
end

function getloss(f, data, loss)
    (x,ygold) = first(data)
    ypred = forw(f, x)
    loss(ypred, ygold)
end


!isinteractive() && !isdefined(:load_only) && Knettest(ARGS)