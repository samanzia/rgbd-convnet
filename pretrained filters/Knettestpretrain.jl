# Implementing a test for the rgb-d dataset
# 148x148 - one layer

using Knet, MAT, ArgParse, CUDNN

@knet function wconvscale(x; out=0, window=0, cinit=Xavier(), o...)
    w = par(; o..., init=cinit, dims=(window, window, 0, out))
	w2 = axpb(w; a=1/80)
    y = conv(w2,x)
end

@knet function convpoolschr(x; out=0, f=relu, cwindow=0, pwindow=0, weights = 0, o...)
	w = par(; o..., init=weights)
    y = conv(w,x)
    z = bias(y; o...)
    r = f(z; o...)
    p = pool(r; o..., window=pwindow)
end


@knet function socher_model(x0; w2 = 0, o...)
    x1 = convpoolschr(x0; out=128, f=relu, cwindow=9, pwindow=10, weights = w2)
    p  = wbf(x1; out=51, f=soft)
end

function Knettest(args=ARGS)
    info("Testing socher's code on rgb-d dataset")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--nbatch"; arg_type=Int; default=128)
        ("--lr"; arg_type=Float64; default=0.0001)
        ("--epochs"; arg_type=Int; default=2)
        ("--gcheck"; arg_type=Int; default=0)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
    
	xtrnFile = matopen("TrainRGBSplit1.mat")
	xtrn = read(xtrnFile,"TrainRGBSplit1")
    xtrn = map(Float32,xtrn)
	println("Loading xtrn")

	xtstFile = matopen("TestRGBSplit1.mat")
	xtst = read(xtstFile,"TestRGBSplit1")
    xtst = map(Float32,xtst)
	println("Loading xtst")
    
	println(maximum(xtrn))
	println(minimum(xtrn))
	println(maximum(xtst))
	println(minimum(xtst))
	
	ytrnFile = matopen("TrainRGBSplit1LabelsMat.mat")
	ytrn = read(ytrnFile,"TrainRGBSplit1LabelsMat")
    ytrn = map(Float32,ytrn)
	println("Loading ytrn")
	

	ytstFile = matopen("TestRGBSplit1LabelsMat.mat")
	ytst = read(ytstFile,"TestRGBSplit1LabelsMat")
    ytst = map(Float32,ytst)
	println("Loading ytst")
	
	file = matopen("filtersResized.mat")
	weights = read(file,"filters")
    weights = map(Float32,weights)
	println("Loading weights")
	
	dtrn = ItemTensor(xtrn, ytrn; batch=nbatch)
    dtst = ItemTensor(xtst, ytst; batch=nbatch)

    socher = FNN(socher_model; w2 = weights)
    setopt!(socher; lr=lr)
    l=zeros(2); m=zeros(2)
    for epoch=1:epochs
        train(socher,dtrn,softloss; losscnt=fill!(l,0), maxnorm=fill!(m,0))
        atrn = 1-test(socher,dtrn,zeroone)
        atst = 1-test(socher,dtst,zeroone)
        println((epoch, atrn, atst, l[1]/l[2], m...))
        gcheck > 0 && gradcheck(socher,dtrn,softloss; gcheck=gcheck)
    end
    return (l[1]/l[2],m...)
end

!isinteractive() && !isdefined(:load_only) && Knettest(ARGS)