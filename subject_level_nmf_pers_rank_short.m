% Full analysis for short range
% - Cluster of meta-rhythms 
% - Extraction of period ranges
% - Re-ordering of meta-behaviour following meta-rhythms clusters
%
% Enea Ceolini, Leiden University

load('./data/ape_padded_and_non_padded_v5.mat')
load('./data/ForCluster.mat', 'Period')
load('./data/perferred_ranks_short_v5.mat')
%% select tensor to use
active_tensor = padded_ape;

%% initialization

n_subs = size(active_tensor, 1);
b_beg = 190;
b_end = 336;
C = colororder;
tt = days(Period);

%%
all_W = cell(n_subs, 1);
all_H = cell(n_subs, 1);
for i = 1:n_subs
    load(sprintf('./staNMFDicts/short/SUB%d/best/best_WH.mat', i))
    all_W{i} = W;
    all_H{i} = full(H);
end

%% find optimal number of clusters

rep_age = cell(length(all_W), 1);
rep_gender = cell(length(all_W), 1);
for i = 1:length(all_W)
    rep_age{i} = ones(size(all_W{i}, 2), 1) .* double(all_ages(i));
    rep_gender{i} = ones(size(all_W{i}, 2), 1) .* double(all_genders(i));
end

all_age_singles = cat(1, rep_age{:})';
all_gender_singles = cat(1, rep_gender{:})';

all_W_singles = cat(2, all_W{:})';


klist = 2:20;
myfunc = @(X,K)(kmeans(X, K, 'Replicates', 10));
myfunc2 = @(X,K)(mdwtcluster(X, 'maxclust', K, 'wname', 'db4').IdxCLU(:,1));
eva_single = evalclusters(zscore(all_W_singles')', myfunc2, 'gap', 'klist', klist);

Gap = eva_single.CriterionValues;
S = eva_single.SE;
right_part  = Gap(2:end) - S(2:end);
left_part = Gap(1:end-1) + S(1:end-1);
cc = left_part >= right_part;

opt_k = find(cc == 1, 1, 'first');

%%
K = opt_k; %eva.OptimalK;

S = mdwtcluster(zscore(all_W_singles')', 'maxclust', K, 'wname', 'db4');

n_scales = size(all_W_singles, 2);

IdxCLU = S.IdxCLU;
mean_from_clusters = zeros(K, n_scales);
for i = 1:K
    subplot(2, K, i)
    imagesc(all_W_singles(IdxCLU(:, 1) == i,:))
    subplot(2, K, i + K)
    plot(days(Period(b_beg:b_end)), mean(all_W_singles(IdxCLU(:, 1) == i,:), 1))
    mean_from_clusters(i, :) = mean(all_W_singles(IdxCLU(:, 1) == i,:), 1);
end

%% order clusters
[~, I] = max(mean_from_clusters, [], 2);
[~, V] = sort(I);
mean_from_clusters = mean_from_clusters(V, :);


%% get binary presence 
n_subs = length(all_W);
slices = [0; cumsum(preferred_ranks)];

classes = IdxCLU(:,1);

binary_presence = zeros(n_subs, K);
for i = 1:n_subs
    cl = classes(slices(i) + 1: slices(i + 1));
    for j = 1:length(cl)
        binary_presence(i, cl(j)) = binary_presence(i, cl(j)) + 1;
    end
end

%%
bp_m = binary_presence(all_genders == 1, :);
ages_m = all_ages(all_genders == 1);

bp_f = binary_presence(all_genders == 2, :);
ages_f = all_ages(all_genders == 2);

[~, If] = sort(ages_f);
[~, Im] = sort(ages_m);

all_bp_sorted = [bp_f(If, :); bp_m(Im, :)];
figure()
subplot(1,2,1)
imagesc(bp_f(If, :))
subplot(1,2,2)
imagesc(bp_f(Im, :))

%%

X = days(Period(b_beg:b_end))';
spacing = round(diff(X)/min(diff(X)));
X_spaced = X(1):min(diff(X)):X(end); % = 0:25:800
figure()

for i = 1:K
    subplot(3,K,i)
    data = all_W_singles(IdxCLU(:,1)==V(i),:);
    data_spaced = repelem(data, 1, spacing([1 1:end]), 1);
    imagesc(X_spaced, [], data_spaced)
    subplot(3,K,i + K)
    shade_iqr(all_W_singles(IdxCLU(:,1)==V(i),:), days(Period(b_beg:b_end)), 'b')
    xlim([days(Period(b_beg)), days(Period(b_end))])
    subplot(3,K,i + K * 2)
    imagesc(binary_presence(:, V(i)))
    colorbar()
end


%% Clustering based on presence absence of clusterized components
% If 2 of the same cluster are prese nt we used the strongest one

slices = [0; cumsum(preferred_ranks)];

classes = IdxCLU(:,1);

feat_matrix = zeros(n_subs, n_scales * K);
for s = 1:n_subs
    cats = classes(slices(s) + 1: slices(s + 1));  % e.g. (5, 5, 4, 1, 3)
    r = length(cats);
    [GC, GCv] = groupcounts(cats);  % count, values 
    if length(GCv) < r  % there are repetitions
        for i = 1:length(GCv)
            if GC(i) == 1  % only one - we put it 
                j = GCv(i);
                original_idx = find(cats == GCv(i));
                j = find(j == V);
                feat_matrix(s, (j - 1) * n_scales + 1: j * n_scales) = zscore(all_W{s}(:, original_idx));
            else
                j = GCv(i);
                original_idxs = find(cats == GCv(i));
                considered = squeeze(all_W{s}(:, original_idxs)); % (147, n)
                [~, I] = max(max(considered, [], 1));
                j = find(j == V);
                feat_matrix(s, (j - 1) * n_scales + 1: j * n_scales) = zscore(considered(:, I));
            end
        end
    else
        % they are all there
        for i = 1:r
            j = cats(i);
            j = find(j == V);
            feat_matrix(s, (j - 1) * n_scales + 1: j * n_scales) = zscore(all_W{s}(:, i));
        end
    end
end

%%
ff = reshape(feat_matrix, n_subs, n_scales, K);
mean_ff = squeeze(mean(ff, 1));
plot(mean_ff)

valid_scales = zeros(K, n_scales);

for IDX = 1:K
    subplot(1,K,IDX)
    a = mean_ff(:, IDX) > prctile(mean_ff(:, IDX), 95);
    valid_scales(IDX, :) = a;
    plot(mean_ff(:, IDX))
    x = find(a == 1)';
    y2 = ones(1, length(x)) * -.5;
    y1 = mean_ff(a==1, IDX)';
    hold on
    patch([x fliplr(x)], [y1 fliplr(y2)], 'b', 'facealpha',0.3, 'edgecolor', 'none')
%     plot(a * max(mean_ff(:, IDX)))
end

figure()
subset_scales = padded_masks(:, b_beg:b_end, :, :);
for IDX = 1:K
    subplot(1,K,IDX)
    subset_scales_any = squeeze(any(subset_scales(:, valid_scales(IDX, :) == 1, :, :), 2));
    imagesc(squeeze(mean(subset_scales_any)))
    colorbar()
end

%% feat mat but for H
slices = [0; cumsum(preferred_ranks)];

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
        % they are all there
        for i = 1:r
            j = cats(i);
            j = find(j == V);
            feat_matrix_H(s, j, :) = all_H{s}(i, :);
        end
    end
end

feat_matrix_H_re = reshape(feat_matrix_H, n_subs, K, 50, 50);
tt2 = days(Period(b_beg:b_end));
for i = 1:K
    subplot(3, K,i)
    plot(days(Period(b_beg:b_end)), mean(feat_matrix(:, (i - 1) * n_scales + 1: i * n_scales), 1))
    hold on
    plot(days(Period(b_beg:b_end)), valid_scales(i, :) * max(mean(feat_matrix(:, (i - 1) * n_scales + 1: i * n_scales), 1)))
    
    title(sprintf("%.1f - %.1f", tt2(find(valid_scales(i, :)==1, 1, 'first')),tt2(find(valid_scales(i, :)==1, 1, 'last'))))
    
    subplot(3, K,i + K)
    imagesc(squeeze(mean(feat_matrix_H_re(:, i, :, :), 1)))
    set(gca, 'YDir','normal')
    colorbar()
    subplot(3, K,i + 2 * K)
    subset_scales_any = squeeze(any(subset_scales(:, valid_scales(i, :) == 1, :, :), 2));
    imagesc(squeeze(mean(subset_scales_any)))
    set(gca, 'YDir','normal')
    colorbar()
end

%% save
save('./data/to_plot_fig3_short_v5', 'classes', 'X', 'V','all_W_singles','all_age_singles', 'all_gender_singles', 'all_ages', 'all_genders', 'binary_presence')
save('./data/to_plot_fig4_short_v5', 'feat_matrix_H_re','valid_scales','feat_matrix', 'subset_scales', 'tt2')