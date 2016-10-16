# for 3-D using transferred weights from VGG

using CUDArt
device(0)

using Knet, MAT, ArgParse, CUDNN, JLD

# conv function for 3-D (takes and returns a 5-D tensor)
@knet function wconv3d(x; cinit=Xavier(), p = 0, o...)
    w = par(; o..., init=cinit)
    return conv(w,x; padding = p)
end

# bias function for 3-D (takes and returns a 5-D tensor)
@knet function bias5(x; binit=Constant(0), o...)
    b = par(; o..., init=binit)
    return b+x
end

# function conv+bias+rectification+pool 
@knet function cbfp3d(x; f=:relu, w=Xavier() , b=constant(0) ,pad=0, o...)
    y = wconv3d(x; o..., cinit=w, p=pad)
    z = bias5(y; o..., binit=b)
    r = f(z; o...)
    return pool(r; o..., window=pwindow)
end

# 3-D defined model
@knet function threeD_model(x0; w11=0, w12=0, b11=0, b12=0)
    x1 = cbfp3d(x0; f=:relu, w=w11, b=b11, pwindow=2, pad=1) #using pretrained 3-D VGG weights
	x2 = cbfp3d(x1; f=:relu, w=12, b=b12, pwindow=2, pad=1)
	x3 = wbf(x2; out=4096, f=:relu,winit=Gaussian(0.0, 0.01))
    return wbf(x3; out=51, f=:soft, winit=Gaussian(0.0,0.01))
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
    
    #reads training and testing files (in matlab format) using the mat package
	xtrnFile = matopen("TrainShuffleSplit1.mat")
	xtrn = read(xtrnFile,"TrainSplit1")
    xtrn = map(Float32,xtrn)
	println(size(xtrn))

	xtstFile = matopen("TestShuffleSplit1.mat")
	xtst = read(xtstFile,"TestSplit1")
	#xtst = xtst[:,:,:,:,1:1000]
    xtst = map(Float32,xtst)
	println(size(xtst))
	
	ytrnFile = matopen("TrainShuffleSplit1LabelsMat.mat")
	ytrn = read(ytrnFile,"TrainSplit1LabelsMat")
    ytrn = map(Float32,ytrn)
	println("Loading ytrn")

	ytstFile = matopen("TestShuffleSplit1LabelsMat.mat")
	ytst = read(ytstFile,"TestSplit1LabelsMat")
    #ytst = ytst[:,1:1000]
	ytst = map(Float32,ytst)
	println("Loading ytst")
	
    # normalize the input to be between 0 and 255
	mtrn = maximum(xtrn);
	xtrn = xtrn/mtrn;
	xtst = xtst/mtrn;
	
    #minibatching
    dtrn = minibatch(xtrn, ytrn, nbatch)
    dtst = minibatch(xtst, ytst, nbatch)
	
    #loading VGG 3-D weights
    file = matopen("vgg2layers3DWeights.mat")

	w1_1 = read(file,"w1_13D")
    w1_1 = map(Float32,w1_1)
	println(size(w1_1))
	
	t1_1 = read(file,"b1_1")
	t1_1 = map(Float32,t1_1)
	b1_1 = reshape(t1_1, 1,1,1,size(t1_1,1),1)
	println(size(w1_1))
	
	w1_2 = read(file,"w1_23D")
    w1_2 = map(Float32,w1_2)
    println(size(w1_2))
	
	t1_2 = read(file,"b1_2")
	t1_2 = map(Float32,t1_2)
	b1_2 = reshape(t1_2, 1,1,1,size(t1_2,1),1)
	println(size(b1_2))

    #compiling and training the 3-D model
    threeD = compile(:threeD_model; w11 = w1_1, b11 = b1_1, w12 = w1_2, b12 = b1_2)

    setp(threeD; lr=lr)
	l=zeros(2); m=zeros(2)
    @time for epoch=1:epochs
        train(threeD,dtrn,softloss;losscnt=fill!(l,0), maxnorm=fill!(m,0))
        atrn = 1-test(threeD,dtrn,zeroone)
        atst = 1-test(threeD,dtst,zeroone)
        println((epoch, atrn, atst))
	end
	
    #save the model for later use
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