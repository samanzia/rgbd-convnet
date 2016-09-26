# Implementing a test for the RGB-d dataset
# 148x148 - one layer

using CUDArt
device(3)

using Knet, MAT, ArgParse, CUDNN


@knet function soft_model(x0)
	return wbf(x0; out = 51, f=:soft)
end

function Knettest(args=ARGS)
    #info("Testing vgg's code on RGB-d dataset")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--nbatch"; arg_type=Int; default=300)
        ("--lr"; arg_type=Float64; default=1.2)
        ("--epochs"; arg_type=Int; default=100)
        ("--gcheck"; arg_type=Int; default=0)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
	
	xtrnFile = matopen("TrainShuffleSplit1Features1.mat")
	xtrn = read(xtrnFile,"TrainSplit1")
    xtrn = map(Float32,xtrn)
	println("Loading xtrn")

	xtstFile = matopen("TestShuffleSplit1Features1.mat")
	xtst = read(xtstFile,"TestSplit1")
    xtst = map(Float32,xtst)
	println(size(xtst))
    
	xdevFile = matopen("DevShuffleSplit1Features1.mat")
	xdev = read(xdevFile,"DevSplit1")
    xdev = map(Float32,xdev)
	println(size(xdev))
	
	ytrnFile = matopen("TrainShuffleSplit1LabelsMatFeatures1.mat")
	ytrn = read(ytrnFile,"TrainSplit1LabelsMat")
    ytrn = map(Float32,ytrn)
	
	ytstFile = matopen("TestShuffleSplit1LabelsMatFeatures1.mat")
	ytst = read(ytstFile,"TestSplit1LabelsMat")
    ytst = map(Float32,ytst)
	
	ydevFile = matopen("DevShuffleSplit1LabelsMatFeatures1.mat")
	ydev = read(ydevFile,"DevSplit1LabelsMat")
	ydev = map(Float32,ydev)
	println(size(ytrn));
	println(size(ytst));
	
	global dtrn = minibatch(xtrn, ytrn, nbatch)
    global dtst = minibatch(xtst, ytst, nbatch)
	global ddev = minibatch(xdev, ydev, nbatch)
	
    global softm = compile(:soft_model)
    
	setp(softm; lr=lr)
	setp(softm; adam=true)

	prevdev = 0;
    l=zeros(2); m=zeros(2)
    for epoch=1:epochs
		
        train(softm,dtrn,softloss; losscnt=fill!(l,0), maxnorm=fill!(m,0))
        atrn = 1-test(softm,dtrn,zeroone)
		atst = 1-test(softm,dtst,zeroone)
		adev = 1-test(softm,ddev,zeroone)
        println((epoch, atrn, adev, atst, l[1]/l[2]))
		if(prevdev < adev)
			JLD.save("bestsoftmax.jld", "model", clean(softm));
			prevdev = adev;
			println("Best Development Performance saved");
		end
        gcheck > 0 && gradcheck(vgg, f->getgrad(f,dtrn,softloss), f->getloss(f,dtrn,softloss); gcheck=gcheck)
    end
	
    return (l[1]/l[2],m...)
end

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
