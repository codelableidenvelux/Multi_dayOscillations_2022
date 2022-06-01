% Subject level cross-validation
% for finding the best rank for NNMF
% 
% Enea Ceolini, Leiden University 

%% loading data
load('../PredNextJID/ape_padded_and_non_padded.mat')

%% pick only the subjects that have at least 180 days of data
%  meaning have a full Periodogram (396 Periods)
valid_long = cellfun(@(x) size(x, 1) == 397, ape_jids);
long_ages = all_ages(valid_long);
long_genders = all_genders(valid_long);
full_scales = ape_jids(1, valid_long);
n_subs = length(full_scales);

full_tensor = zeros(n_subs, 396, 2500);
for i = 1:n_subs
    full_tensor(i, :, :) = reshape(full_scales{i}(1:396, :, :), 396, 2500);
end

%% Crosso-validation

n_subs = size(full_tensor, 1);
n_scales = size(full_tensor, 2);
b_beg = 190;  % 2.2  days
b_end = 390;  % 70.5 days

n_ranks = 15;
repetitions = 20;
train_err = zeros(n_subs, n_ranks, repetitions);
test_err = zeros(n_subs, n_ranks, repetitions);
split = 0.9;

parfor IDX = 1:n_subs
    for r = 1:n_ranks
        for rep = 1:repetitions
            
            fprintf("SUB %d - rank %d - rep %d\n", IDX,  r, rep)
            
            % slicing in the 'long' range
            m_a = squeeze(full_tensor(IDX, :, :));
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