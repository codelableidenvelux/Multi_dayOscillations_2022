% Subject level cross-validation
% for finding the best rank for NNMF
% 
% Enea Ceolini, Leiden University 

%% load data
load('../PredNextJID/ape_padded_and_non_padded.mat')

%% Cross-validation

n_subs = size(padded_ape, 1);
n_scales = size(padded_ape, 2);
b_beg = 190;  % 2.2  days
b_end = 336;  % 27.7 days

n_ranks = 15;
repetitions = 20;
train_err = zeros(n_subs, n_ranks, repetitions);
test_err = zeros(n_subs, n_ranks, repetitions);
split = 0.9;

parfor IDX = 1:n_subs

    for rep = 1:repetitions
        for r = 1:n_ranks

            fprintf("SUB %d - rank %d - rep %d\n", IDX,  r, rep)
            
            % slicing in the 'short' range
            m_a = reshape(squeeze(padded_ape(IDX, :, :, :)), n_scales, 2500);
            m_a = m_a(b_beg:b_end, :);
            
            % nan-guard
            m_a(isnan(m_a)) = 0;
            
            % make it non-negative
            mm = min(m_a, [], 1);
            m_a = m_a - min(mm);
            
            % mask values to keep test out
            mask = rand(size(m_a)) > (1 - split);
            masked_a = m_a .* mask;
            
            % find initialization matrices
            opt = statset('MaxIter',100);
            [W0, H0] = nnmf(masked_a, r,'Replicates',100, 'Options',opt, 'Algorithm','mult');
            
            % factorization 
            opt = statset('Maxiter',1000);
            [W, H] = nnmf(masked_a, r,'W0',W0,'H0',H0, 'Options',opt, 'Algorithm','als', 'Replicates',100);
            
            % reconstruction
            recon_a = W * H;
            
            % calculate errors
            train_err(IDX, r, rep) = sum(((recon_a - m_a) .* mask) .^ 2, [1, 2]) / sum(mask, [1,2]);
            test_err(IDX, r, rep) = sum(((recon_a - m_a) .* ~mask) .^ 2, [1, 2]) / sum(~mask, [1,2]);
            
        end
    end
end

%% find best rank based on best test error

te = mean(test_err, 3);
tr = mean(train_err, 3);

[~, r] = min(te, [], 2);