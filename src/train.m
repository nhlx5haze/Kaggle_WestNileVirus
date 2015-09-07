addpath(genpath("./DeepLearnToolbox-master/"));
tx = csvread('preprocesstrain.csv');
tx = tx(1:round(size(tx,1)/20)*20,:);

opts.numepochs          = 60;                        %  Number of full sweeps through data
opts.batchsize          = 20;                        %  Take a mean gradient step over this many samples
opts.plot               = 0;
ensSize = 24;

ty = tx(:,1);
ty(isnan(ty)) = 0;
ty = [ty ~ty];
tx(:,[1:3 end]) = [];

[tx, mu, sigma] = zscore(tx);

tlist = {};
nnlist = {};

for i=1:ensSize
    rand('state',i*1234);
    
    randBaggingAttribute = randperm(size(tx,2));
    tx2 = tx(:,randBaggingAttribute(1:15));
    
    if mod(i,2) == 0
        nn = nnsetup([size(tx2,2) 50+randi(50) size(ty,2)]);
    elseif mod(i,2) == 1
        nn = nnsetup([size(tx2,2) 50+randi(50) 25+randi(25) size(ty,2)]);
    end
    
    nn.randBaggingAttribute = randBaggingAttribute;
    nn.output               = 'softmax';                   %  use softmax output
    nn = nntrain(nn, tx2, ty, opts);                %  nntrain takes validation set as last two arguments (optionally)
    nn.e = [];
    nn.a = {};
    
    nnlist{end+1} = nn;
    disp(['training ensemble ' num2str(i) '/24']);
    fflush(stdout); 
end

nninfo = struct;

nninfo.mu = mu;
nninfo.sigma = sigma;
nninfo.nnlist = nnlist;

save('modelNNOctave.dat', 'nninfo');



