# Implementing a test for the RGB-d dataset
# 148x148 - one layer

using CUDArt
device(1)

using Knet, MAT, ArgParse, CUDNN, JLD

@knet function cbf(x;f=:relu, w = Xavier(), b = Constant(0) , p= 1, out = 0, o...)
    v = par(; o... , init = w, out=out)
	y = conv(v,x; padding = p, mode = CUDNN_CROSS_CORRELATION)
    z = bias4(y; binit=b, outDim = out, o...)
    return f(z; o...)
end

@knet function bias4(x; binit=Constant(0), outDim = 0, o...)
    b = par(; o..., init=binit)
    return b+x
end

@knet function cb(x;w = Xavier(), b = Constant(0) , p= 0, s = 1, o...)
    v = par(; o... , init = w)
	y = conv(v,x; padding = p, stride = s, mode = CUDNN_CROSS_CORRELATION)
    return bias4(y; binit =b,o...)
end

@knet function cbfp(x; f=:relu, cwindow=0, pwindow=0, o...)
    y = wconv(x; o..., window=cwindow)
    z = bias4(y; o...)
    return f(z; o...)
    
end


@knet function vgg_model(x0; w11=0,w12=0,w21=0,w22=0,w31=0,w32=0,w33=0,w41=0,w42=0,w43=0,w51=0,w52=0,w53=0,w66=0,w88=0,b11=0,b12=0,b21=0,b22=0,b31=0,b32=0,b33=0,b41=0,b42=0,b43=0,b51=0,b52=0,b53=0,b66=0,b88=0)

    x1 = cbf(x0; w = w11, b = b11, out=64)
	x2 = cbf(x1; w = w12, b = b12, out=64)
	x3 = pool(x2; window = 2)
	#x33 = drop(x3;pdrop=0.5)
	x4 = cbf(x3; w = w21, b = b21, out=128)
	x5 = cbf(x4; w = w22, b = b22, out=128)
	x6 = pool(x5; window = 2)
	#x66 = drop(x6;pdrop=0.5)
	x7 = cbf(x6; w = w31, b = b31, out=256)
	x8 = cbf(x7; w = w32, b = b32, out=256)
	x9 = cbf(x8; w = w33, b = b33, out=256)
	x10 = pool(x9; window = 2)
	#x101 = drop(x10;pdrop=0.5)
	x11 = cbf(x10; w = w41, b = b41, out=512)
	x12 = cbf(x11; w = w42, b = b42, out=512)
	x13 = cbf(x12; w = w43, b = b43, out=512)
	x14 = pool(x13; window = 2)
	#x144 = drop(x14;pdrop=0.5)
	x15 = cbf(x14; w = w51, b = b51, out=512)
	x16 = cbf(x15; w = w52, b = b52, out=512)
	x17 = cbf(x16; w = w53, b = b53, out=512)
	x18 = pool(x17; window = 2)	
	x19 = drop(x18;pdrop=0.5)
    x20 = cbf(x19;w = w66, b=b66, f=:relu, p=0)
	x21 = drop(x20;pdrop=0.5)
	return wbf(x21;winit = w88, binit=b88, out = 51, f=:soft)
end

function Knettest(args=ARGS)
    #info("Testing vgg's code on RGB-d dataset")
    s = ArgParseSettings()
    @add_arg_table s begin
        ("--seed"; arg_type=Int; default=42)
        ("--nbatch"; arg_type=Int; default=70)
        ("--lr"; arg_type=Float64; default=0.00001)
        ("--epochs"; arg_type=Int; default=2)
        ("--gcheck"; arg_type=Int; default=0)
    end
    isa(args, AbstractString) && (args=split(args))
    opts = parse_args(args, s)
    println(opts)
    for (k,v) in opts; @eval ($(symbol(k))=$v); end
    seed > 0 && setseed(seed)
	
	xtrnFile = matopen("TrainShuffleSplit1.mat")
	xtrn = read(xtrnFile,"TrainSplit1")
    xtrn = map(Float32,xtrn)
	println(size(xtrn))

	xtstFile = matopen("TestShuffleSplit1.mat")
	xtst = read(xtstFile,"TestSplit1")
    xtst = map(Float32,xtst)
	println(size(xtst))
    
	#xdevFile = matopen("DevShuffleSplit1.mat")
	#xdev = read(xdevFile,"DevSplit1")
    #xdev = map(Float32,xdev)
	#println(size(xdev))
	
	ytrnFile = matopen("TrainShuffleSplit1LabelsMat.mat")
	ytrn = read(ytrnFile,"TrainSplit1LabelsMat")
    ytrn = map(Float32,ytrn)
	println(size(ytrn))
	
	ytstFile = matopen("TestShuffleSplit1LabelsMat.mat")
	ytst = read(ytstFile,"TestSplit1LabelsMat")
    ytst = map(Float32,ytst)
	println(size(ytst))
	
	#ydevFile = matopen("DevShuffleSplit1LabelsMat.mat")
	#ydev = read(ydevFile,"DevSplit1LabelsMat")
	#ydev = map(Float32,ydev)

	
	file = matopen("vgg-verydeep-16.mat")
	
	file1 = matopen("w13D.mat")
	w1_1 = read(file1,"w13D")
    w1_1 = map(Float32,w1_1)
	
	
	t1_1 = read(file,"b1_1")
	t1_1 = map(Float32,t1_1)
	b1_1 = reshape(t1_1, 1,1,size(t1_1,1),1)
	println(size(w1_1))
	
	w1_2 = read(file,"w1_2")
    w1_2 = map(Float32,w1_2)
	
	t1_2 = read(file,"b1_2")
	t1_2 = map(Float32,t1_2)
	b1_2 = reshape(t1_2, 1,1,size(t1_2,1),1)
	println(size(b1_2))
	
	w2_1 = read(file,"w2_1")
    w2_1 = map(Float32,w2_1)
	
	t2_1 = read(file,"b2_1")
	t2_1 = map(Float32,t2_1)
	b2_1 = reshape(t2_1, 1,1,size(t2_1,1),1)
	println(size(b2_1))
	
	w2_2 = read(file,"w2_2")
    w2_2 = map(Float32,w2_2)
	
	t2_2 = read(file,"b2_2")
	t2_2 = map(Float32,t2_2)
	b2_2 = reshape(t2_2, 1,1,size(t2_2,1),1)
	println(size(b2_2))
	
	w3_1 = read(file,"w3_1")
    w3_1 = map(Float32,w3_1)
	
	t3_1 = read(file,"b3_1")
	t3_1 = map(Float32,t3_1)
	b3_1 = reshape(t3_1, 1,1,size(t3_1,1),1)
	println(size(w3_1))
	
	w3_2 = read(file,"w3_2")
    w3_2 = map(Float32,w3_2)
	
	t3_2 = read(file,"b3_2")
	t3_2 = map(Float32,t3_2)
	b3_2 = reshape(t3_2, 1,1,size(t3_2,1),1)
	println(size(w3_2))
	
	w3_3 = read(file,"w3_3")
    w3_3 = map(Float32,w3_3)
	
	t3_3 = read(file,"b3_3")
	t3_3 = map(Float32,t3_3)
	b3_3 = reshape(t3_3, 1,1,size(t3_3,1),1)
	println(size(w3_3))
	
	w4_1 = read(file,"w4_1")
    w4_1 = map(Float32,w4_1)
	
	t4_1 = read(file,"b4_1")
	t4_1 = map(Float32,t4_1)
	b4_1 = reshape(t4_1, 1,1,size(t4_1,1),1)
	println(size(w4_1))
	
	w4_2 = read(file,"w4_2")
    w4_2 = map(Float32,w4_2)
	
	t4_2 = read(file,"b4_2")
	t4_2 = map(Float32,t4_2)
	b4_2 = reshape(t4_2, 1,1,size(t4_2,1),1)
	println(size(w4_2))
	
	w4_3 = read(file,"w4_3")
    w4_3 = map(Float32,w4_3)
	
	t4_3 = read(file,"b4_3")
	t4_3 = map(Float32,t4_3)
	b4_3 = reshape(t4_3, 1,1,size(t4_3,1),1)
	println(size(w4_3))
	
	w5_1 = read(file,"w5_1")
    w5_1 = map(Float32,w5_1)
	
	t5_1 = read(file,"b5_1")
	t5_1 = map(Float32,t5_1)
	b5_1 = reshape(t5_1, 1,1,size(t5_1,1),1)
	println(size(w5_1))
	
	w5_2 = read(file,"w5_2")
    w5_2 = map(Float32,w5_2)
	
	t5_2 = read(file,"b5_2")
	t5_2 = map(Float32,t5_2)
	b5_2 = reshape(t5_2, 1,1,size(t5_2,1),1)
	println(size(w5_2))
	
	w5_3 = read(file,"w5_3")
    w5_3 = map(Float32,w5_3)
	
	t5_3 = read(file,"b5_3")
	t5_3 = map(Float32,t5_3)
	b5_3 = reshape(t5_3, 1,1,size(t5_3,1),1)
	println(size(w5_3))

    w6 = read(file, "w6")
	w6 = map(Float32,w6)
	
	b6 = read(file, "b6")
	b6 = map(Float32,b6)
	b6 = reshape(b6, 1,1,size(b6,1),1)
	println(size(w6))
	
	w8 = load("softmaxweights.jld", "w8");
	b8 = load("softmaxweights.jld","b8");
	
    println(size(w8))
	println(size(b8))
	
	
	global dtrn = minibatch(xtrn, ytrn, nbatch)
    global dtst = minibatch(xtst, ytst, nbatch)
	#global ddev = minibatch(xdev, ydev, nbatch)
	
    global vgg = compile(:vgg_model; w11 = w1_1, b11 = b1_1, w12 = w1_2, b12 = b1_2, w21 = w2_1, b21 = b2_1, w22 = w2_2, b22 = b2_2, w31 = w3_1, b31 = b3_1, w32 = w3_2, b32 = b3_2, w33 = w3_3, b33 = b3_3, w41 = w4_1, b41 = b4_1, w42 = w4_2, b42 = b4_2, w43 = w4_3, b43 = b4_3, w51 = w5_1, b51 = b5_1, w52 = w5_2, b52 = b5_2, w53 = w5_3, b53 = b5_3, w66 = w6, b66 = b6, w88=w8, b88=b8)

	setp(vgg; lr=lr)
	setp(vgg; adam=true)
	
    l=zeros(2); m=zeros(2)
    for epoch=1:epochs
		tic();
        train(vgg,dtrn,softloss; losscnt=fill!(l,0), maxnorm=fill!(m,0))
        atrn = 1-test(vgg,dtrn,zeroone)
		atst = 1-test(vgg,dtst,zeroone)
		#adev = 1-test(vgg,ddev,zeroone)
        println((epoch, atrn, atst, l[1]/l[2]))
		println(toc())
        gcheck > 0 && gradcheck(vgg, f->getgrad(f,dtrn,softloss), f->getloss(f,dtrn,softloss); gcheck=gcheck)
    end
	JLD.save("VGG1.jld", "model", clean(vgg));
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
