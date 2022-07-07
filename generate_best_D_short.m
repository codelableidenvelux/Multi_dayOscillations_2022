function generate_best_D_short_v5(IDX)

% Generate the best set of meta-behaviours/meta-rhytms for each subject
% Given the best rank coming from the cross-validation, we do the following
% to find the more stable decomposition:
% - Repeat the decomposition 1000 times
% - Calculate the pairwise correlation coefficient for all decompositions
% - Calculate the median correlation coefficient for each decomposition
% - Select as most stable the decomposition with the highest correlation
% coefficient.
%
% Note: this code requires SPAMS for the NNMF.
% Enea Ceolini, Leiden University

% load subject level cross-validation
load('./data/ape_padded_and_non_padded_v5.mat', 'padded_ape')
load('./data/perferred_ranks_short_v5.mat', 'preferred_ranks')

% replace the following two paths with your SPAMS installation paths
addpath('../spams-matlab-v2.6/');
addpath('../spams-matlab-v2.6/build/');

%%

b_beg = 190;
b_end = 336;

K = preferred_ranks(IDX);
masked_a = reshape(squeeze(padded_ape(IDX, :, :, :)), n_scales, 2500);
masked_a = masked_a(b_beg:b_end, :);

% nan-guard
masked_a(isnan(masked_a)) = 0;

mm = min(masked_a, [], 1);
masked_a = masked_a - min(mm);

X = masked_a;
[~,n] = size(X);

%%
numRep = 1000;
lambda = 0;         % sparsity control for the coefficients alpha
gamma1 = 0; % sparsity control on the dictionary patterns

D0 = dictLearnInit(X,K); % randomly select K columns
% from the data matrix X

path = ['./staNMFDicts/short/SUB',num2str(IDX),'/best/K=',num2str(K),'/'];
mkdir(path);

param.mode = 2;
param.K=K;
param.lambda=lambda;
param.numThreads=-1;
param.batchsize=min(1024,n);
param.posD = true;   % positive dictionary
param.iter = 500;  % number of iteration
param.modeD = 0;
param.verbose = 0; % print out update information?
param.posAlpha = 1; % positive coefficients
param.gamma1 = gamma1; % penalizing parameter on the dictionary patterns
param.D = D0; % set initial values

% For each fixed dictionary K, repeat NMF for 1000 times, each with a different initial value

for i = 1:numRep

    D0 = dictLearnInit(X, K);
    param.D = D0;
    tic
    D = mexTrainDL(X, param);
    fprintf("%d took %.2f s\n", i, toc);
    toc
    % save the dictionary for future use
    save([path,'rep',num2str(i),'Dict.mat'],'D');

end

%% now find the best Dictionary based on stability

loadPath = [path,'rep1Dict.mat'];
load(loadPath, 'D');
d = size(D,1);

Dhat = zeros(d,K,numRep);
for L = 1:numRep
    loadPath = [path,'rep',num2str(L),'Dict.mat'];
    load(loadPath, 'D');
    Dhat(:,:,L) = D;
end

distMat = zeros(numRep,numRep);

for q = 1:numRep
    for p = q:numRep
        CORR = corr(Dhat(:,:,q),Dhat(:,:,p));
        distMat(q,p) = amariMaxCorr(CORR);
        distMat(p,q) = distMat(q,p);
    end
end
save([path,'distMatrixDictCorr.mat'],'distMat');

estStability = median(distMat, 1);

[~, idx_best_D] = max(estStability);

loadPath = [path,'rep',num2str(idx_best_D),'Dict.mat'];
load(loadPath, 'D');

H = mexLasso(X,D,param);

path = ['./staNMFDicts/short/SUB',num2str(IDX),'/best/best_WH.mat'];
W = D;
save(path, 'W', 'H')


end
