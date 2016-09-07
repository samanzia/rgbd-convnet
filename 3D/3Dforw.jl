# For extracting features from already trained models (only does a forward pass and
# returns the features from the second last layer) 

using CUDArt
device(0)

using Knet, MAT, ArgParse, CUDNN, JLD

function Knettest(args=ARGS)
    #info("Testing 3-D on RGB-d dataset")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--nbatch"; arg_type=Int; default=32)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
	
    #reading dataset
	xtrnFile = matopen("TrainSplit1.mat")
	xtrn = read(xtrnFile,"TrainSplit1")
    xtrn = map(Float32,xtrn)
	println("Loading xtrn")

	xtstFile = matopen("TestSplit1.mat")
	xtst = read(xtstFile,"TestSplit1")
    xtst = map(Float32,xtst)
	println(size(xtst))
    
	#xdevFile = matopen("DevRGBSplit1.mat")
	#xdev = read(xdevFile,"DevRGBSplit1")
    #xdev = map(Float32,xdev)
	#println(size(xdev))
	
	ytrnFile = matopen("TrainSplit1LabelsMat.mat")
	ytrn = read(ytrnFile,"TrainSplit1LabelsMat")
    ytrn = map(Float32,ytrn)
	
	ytstFile = matopen("TestSplit1LabelsMat.mat")
	ytst = read(ytstFile,"TestSplit1LabelsMat")
    ytst = map(Float32,ytst)
	
	#ydevFile = matopen("DevRGBSplit1LabelsMat.mat")
	#ydev = read(ydevFile,"DevRGBSplit1LabelsMat")
	#ydev = map(Float32,ydev)

	#normalizing to makes values lie between 0 and 255
	mtrn = maximum(xtrn);
	xtrn = xtrn/mtrn;
	xtst = xtst/mtrn;

	dim = 4096;

	global threeD = JLD.load("threeD.jld", "model")  
	xtrnsize =size(xtrn,5)
	xtrnzerosize = xtrnsize + nbatch - xtrnsize%nbatch;
	println(xtrnzerosize)
	xtrnZeroPad = zeros(size(xtrn,1),size(xtrn,2),size(xtrn,3),size(xtrn,4),xtrnzerosize)
	for x=1:size(xtrn,5)
		xtrnZeroPad[:,:,:,:,x] = xtrn[:,:,:,:,x]
	end
	

	xtrnZeroPad = map(Float32, xtrnZeroPad)
	xtrnFeatures = zeros(dim,xtrnzerosize)
	for item = 1:nbatch:size(xtrn,5)
		ypred = forw(threeD, xtrnZeroPad[:,:,:,:,item:item+nbatch-1])
		xtrnFeatures[:,item:item+nbatch-1] = to_host(threeD.reg[17].out)
	end
	
	xtrnFeatures = xtrnFeatures[:,1:xtrnsize]
	
	filetrn = matopen("TrainRGBSplit1Features3D.mat", "w")
	write(filetrn, "TrainRGBSplit1Features3D", xtrnFeatures)
	close(filetrn)
	
	xtrnFeatures = 0;
	xtrnZeroPad = 0;
	gc();
	
	xtstsize =size(xtst,5)
	xtstzerosize = xtstsize + nbatch - xtstsize%nbatch;#map(Int64,(xtstsize/nbatch +1)*nbatch);
	println(xtrnzerosize)
	xtstZeroPad = zeros(size(xtst,1),size(xtst,2),size(xtst,3),size(xtst,4),xtstzerosize)
	for x=1:size(xtst,5)
		xtstZeroPad[:,:,:,:,x] = xtst[:,:,:,:,x]
	end
	
	xtstZeroPad = map(Float32, xtstZeroPad)
	xtstFeatures = zeros(dim,xtstzerosize)
    for item = 1:nbatch:size(xtst,5)
		ypred = forw(threeD, xtstZeroPad[:,:,:,:,item:item+nbatch-1])
		xtstFeatures[:,item:item+nbatch-1] = to_host(threeD.reg[17].out)
	end
	
	xtstFeatures = xtstFeatures[:,1:xtstsize]
	
	filetst = matopen("TestRGBSplit1Features3D.mat", "w")
	write(filetst, "TestRGBSplit1Features3D", xtstFeatures)
	close(filetst)
end

#helper methods
function train(f, data, loss; losscnt=nothing, maxnorm=nothing)
    for (x,ygold) in data
        ypred = forw(f, x)
		#println(to_host(size(ypred)))
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
