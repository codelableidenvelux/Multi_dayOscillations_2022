% Find the best rank for each subject based on the cross validation
% This is run after the subjet_level_CV_*
%
% Enea Ceolini, Leiden University

%% long
n_subs = 218;
ranks = 3:14;
preferred_ranks = zeros(n_subs,1);
for IDX = 1:n_subs
    load(['./staNMFDicts/long/SUB', num2str(IDX), '/CV/train_test_CV.mat'])
    [~, best_rank_idx] = min(mean(test_err(:, 1:100), 2));
    preferred_ranks(IDX) = ranks(best_rank_idx);
end

save('./data/perferred_ranks_long_v5.mat', 'preferred_ranks')

%% short
n_subs = 401;
ranks = 3:14;
preferred_ranks = zeros(n_subs,1);
for IDX = 1:n_subs
    load(['./staNMFDicts/short/SUB', num2str(IDX), '/CV/train_test_CV.mat'])
    [~, best_rank_idx] = min(mean(test_err(:, 1:100), 2));
    preferred_ranks(IDX) = ranks(best_rank_idx);
end

save('./data/perferred_ranks_short_v5.mat', 'preferred_ranks')