% Full analysis for long range
% - Subject-level NNMF with preferred rank
% - Cluster of meta-rhythms 
% - Extraction of period ranges
% - Re-ordering of meta-behaviour following meta-rhythms clusters
%
% Enea Ceolini, Leiden University

%% load data

load('../PredNextJID/ape_padded_and_non_padded.mat')
load('full_CV_all_long_sorted.mat')
load('ForCluster.mat', 'Period')

%% pick only the subjects that have at least 180 days of data
%  meaning have a full Periodogram (396 Periods)
valid_long = cellfun(@(x) size(x, 1) == 397, ape_jids);
long_ages = all_ages(valid_long);
long_genders = all_genders(valid_long);
long_ids = 1:344;
long_ids = long_ids(valid_long);
full_scales = ape_jids(1, valid_long);
n_subs = length(full_scales);

full_tensor = zeros(n_subs, 396, 2500);
for i = 1:n_subs
    full_tensor(i, :, :) = reshape(full_scales{i}(1:396, :, :), 396, 2500);
end
%% select tensor to use
active_tensor = full_tensor;

%% initialization

n_subs = size(active_tensor, 1);
n_tot_scales = size(active_tensor, 2);
b_beg = 190;  % 2.2  days
b_end = 390;  % 70.5 days
tt = days(Period);

all_W = cell(n_subs, 1);
all_H = cell(n_subs, 1);

%% Subject-level factorization with best rank

% find preferred rank based on test error during cross-validation
m_test_e = mean(test_err, 3);
[~, preferred_rank] = min(m_test_e, [], 2);

parfor IDX = 1:n_subs
    r = preferred_rank(IDX)
    fprintf("Running %d (rank %d)\n", IDX, r)
        
    % slice to get `long` range
    masked_a = squeeze(active_tensor(IDX, :, :));
    masked_a = masked_a(b_beg:b_end, :);
    
    % nan-guard
    masked_a(isnan(masked_a)) = 0;
    
    % make it non-negative
    mm = min(masked_a, [], 1);
    masked_a = masked_a - mm;
    
    % find initialization matrices
    opt = statset('MaxIter',100);
    [W0, H0] = nnmf(masked_a, r,'Replicates',100, 'Options',opt, 'Algorithm','mult');
    
    % factorization
    opt = statset('Maxiter',1000);
    [W, H] = nnmf(masked_a, r,'W0',W0,'H0',H0, 'Options',opt, 'Algorithm','als', 'Replicates',100);

    % sort meta-rhythms and meta-behaviours based on peak position 
    [~, I] = max(W, [], 1);
    [~, V] = sort(I);
    W = W(:, V);
    H = H(V, :);
    all_W{IDX} = W;
    all_H{IDX} = H;

end

%% save result
save('subject_level_nmf_long_preferred_ranks_sorted_v2', 'all_H', 'all_W', 'preferred_rank')       

%% find number of clusters in the meta-rhytms across the population 

% pull all meta-rhytms together
all_W_singles = cat(2, all_W{:})';

% annotate each meta-rhythm with age and gender
rep_age = cell(length(all_W), 1);
rep_gender = cell(length(all_W), 1);
rep_ids = cell(length(all_W), 1);
for i = 1:length(all_W)
    rep_age{i} = ones(size(all_W{i}, 2), 1) .* double(long_ages(i));
    rep_gender{i} = ones(size(all_W{i}, 2), 1) .* double(long_genders(i));
    rep_ids{i} = ones(size(all_W{i}, 2), 1) .* double(long_ids(i));
end

all_age_singles = cat(1, rep_age{:})';
all_gender_singles = cat(1, rep_gender{:})';
all_ids_singles = cat(1, rep_ids{:})';

% find best number of clusters based on Silhouette
% here we use 1-dimensional DWT clustering
klist = 2:20;
myfunc2 = @(X,K)(mdwtcluster(X, 'maxclust', K, 'wname', 'db4').IdxCLU(:,1));
eva_single = evalclusters(zscore(all_W_singles')', myfunc2, 'Silhouette', 'klist', klist);

%% do the clustering with the optimal number of clusters

K = eva_single.OptimalK;

S = mdwtcluster(zscore(all_W_singles')', 'maxclust', K, 'wname', 'db4');

n_scales = size(all_W_singles, 2);

IdxCLU = S.IdxCLU;

% find the mean of each cluster
mean_from_clusters = zeros(K, n_scales);
for i = 1:K
    mean_from_clusters(i, :) = mean(all_W_singles(IdxCLU(:, 1) == i,:), 1);
end

%% sort clusters based on the peak location of the mean of each cluster
[~, I] = max(mean_from_clusters, [], 2);
[~, V] = sort(I);
mean_from_clusters = mean_from_clusters(V, :);

%% binary presence
%  for each subject we count the number meta-rhythms for each cluster
%  each subject can have none or multiple meta-rythms in the same cluster

n_subs = length(all_W);
slices = [0; cumsum(preferred_rank)];

classes = IdxCLU(:,1);

binary_presence = zeros(n_subs, K);
for i = 1:n_subs
    cl = classes(slices(i) + 1: slices(i + 1));
    for j = 1:length(cl)
        binary_presence(i, cl(j)) = binary_presence(i, cl(j)) + 1;
    end
end

%% Find period ranges for each cluster peak

valid_scales = zeros(K, n_scales);

for IDX = 1:K
    a = mean_from_clusters(IDX, :) > prctile(mean_from_clusters(IDX, :), 95);
    valid_scales(IDX, :) = a;
end

% masks in the `short` range
subset_scales = padded_masks(:, b_beg:b_end, :, :);

%% extract meta-behaviour matrix for all subjects
slices = [0; cumsum(preferred_rank)];

classes = IdxCLU(:,1);

feat_matrix_H = zeros(n_subs, K, 2500);
for s = 1:n_subs
%     fprintf("SUB - %d\n", s)
    cats = classes(slices(s) + 1: slices(s + 1));  % e.g. (5, 5, 4, 1, 3)
    r = length(cats);
    [GC, GCv] = groupcounts(cats);  % count, values 
    if length(GCv) < r  % there are repetitions
        for i = 1:length(GCv)
            if GC(i) == 1  % only one - we put it 
                j = GCv(i);
                original_idx = find(cats == GCv(i));
                j = find(j == V);
                feat_matrix_H(s, j, :) = all_H{s}(original_idx, :);
            else
                j = GCv(i);
                original_idxs = find(cats == GCv(i));
                considered = squeeze(all_H{s}(original_idxs, :)); % (147, n)
                [~, I] = max(max(considered, [], 2));
                j = find(j == V);
                feat_matrix_H(s, j, :) = sum(considered, 1);
            end
        end
    else
        % they are all there: one meta-rhytm for each cluster
        for i = 1:r
            j = cats(i);
            j = find(j == V);
            feat_matrix_H(s, j, :) = all_H{s}(i, :);
        end
    end
end

% reshape meta-bheaviour to (n_subs, #-clusters, 50, 50)
feat_matrix_H_re = reshape(feat_matrix_H, n_subs, K, 50, 50);
tt2 = days(Period(b_beg:b_end));