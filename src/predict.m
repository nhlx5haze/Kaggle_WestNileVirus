addpath(genpath("./DeepLearnToolbox-master/"));
addpath(genpath("./src/"));
datatest = csvread('preprocesstest.csv');
vx = datatest(:,4:end-1);


nninfo = load('modelNNOctave.dat');
nnlist = nninfo.nninfo.nnlist;

vx = normalize(vx, nninfo.nninfo.mu, nninfo.nninfo.sigma);
tlist = {};

for i=1:length(nnlist)
    nn = nnlist{i};
    
    vx2 = vx(:,nn.randBaggingAttribute(1:15));
    nn.testing = 1;      
    nn = nnff(nn,vx2, 0);
    tlist{i} = nn.a{end}(:,1);
    
    disp(['testing ensemble ' num2str(i) '/24']);
    fflush(stdout); 
end

test_probs = mean(cat(2,[],tlist{:}),2);
test_probs = adjustProba(test_probs, datatest );
test_probs_lasagne = csvread('submissionlasagna.tmp',1,0);
makeResultFile((test_probs + test_probs_lasagne(:,2).*0.2)./1.2+0.1, 'submission.csv');




