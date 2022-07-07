function sub_lvl_CV_short_v5(IDX)
% Subject level cross-validation
% for finding the best rank for NNMF
% 
% Enea Ceolini, Leiden University 

fprintf("Received %d\n", IDX)
% subject level cross-validation
load('./data/ape_padded_and_non_padded_v5.mat', 'padded_ape')

% replace the following two paths with your SPAMS installation paths
addpath('../spams-matlab-v2.6/');
addpath('../spams-matlab-v2.6/build/');

%% intiialization
n_scales = size(padded_ape, 2);
b_beg = 190;
b_end = 336;

numPatterns = 3:14; % dictionary sizes
numRep = 100;
lambda = 0;         % sparsity control for the coefficients alpha
gamma1 = 0;         % sparsity control on the dictionary patterns

train_err = zeros(length(numPatterns), numRep);
test_err = zeros(length(numPatterns), numRep);

split = 0.9;

masked_a = reshape(squeeze(padded_ape(IDX, :, :, :)), n_scales, 2500);
masked_a = masked_a(b_beg:b_end, :);

% nan-guard
masked_a(isnan(masked_a)) = 0;

mm = min(masked_a, [], 1);
masked_a = masked_a - min(mm);

X = masked_a;
[~,n] = size(X);

for k = 1:length(numPatterns)

    K = numPatterns(k);
    D0 = dictLearnInit(X,K); % randomly select K columns
    % from the data matrix X

    path = ['./staNMFDicts/short/SUB',num2str(IDX),'/CV/K=',num2str(K),'/'];
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

    % For each fixed dictionary K, repeat NMF for 100 times, each with a different initial value
    for i = 1:numRep
        fprintf("SUB %d - RANK %d - rep %d \n", IDX, K, i);
        mask = rand(size(X)) > (1 - split);
        m_a = X .* mask;

        D0 = dictLearnInit(m_a, K);
        param.D = D0;
        tic
        D = mexTrainDL(m_a, param);
        toc
        % save the dictionary for future use
        save([path,'rep',num2str(i),'Dict.mat'],'D');

        for kk = 1:k
            D(:,kk) = D(:,kk)/max(D(:,kk));
        end

        % nonnegative least squares
        alpha = mexLasso(m_a, D, param);
        recon_a = D * alpha;

        train_err(k, i) = sum(((recon_a - X) .* mask) .^ 2, [1, 2]) / sum(mask, [1,2]);
        test_err(k, i) = sum(((recon_a - X) .* ~mask) .^ 2, [1, 2]) / sum(~mask, [1,2]);

    end
end

path = ['./staNMFDicts/short/SUB',num2str(IDX),'/CV/'];
save([path,'train_test_CV.mat'], 'train_err', 'test_err');


end
