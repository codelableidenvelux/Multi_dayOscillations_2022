% Boostrap Spectral Coherence for `short` Periods
% - For each subject and each scale find the best JID bin in the biggest
% cluster
% - We pull all of the best JID bins together
% - We create pseudo subjects by doing a block boostrap across all best JOD
% bins
%
% Enea Ceolini, Leiden University

%% load data
load('/media/gamma/JIDcompiledHourly_sel.mat')
load('../PredNextJID/ape_padded_and_non_padded_v2.mat')
load('full_CV_all_short_sorted.mat')
load('ForCluster.mat', 'Period')
load('to_plot_fig3_short.mat')
load('to_plot_fig4_short.mat')

%% initialization
b_beg = 190;
b_end = 336;

% this is due to python loading a MATLAB tensor of size X, Y, Z in the 
% order Z, Y, X
re_padded_clu = reshape(padded_clu, 344, 397, 50, 50);
p_re_padded_clu = permute(re_padded_clu, [1,2,4,3]);
padded_clu = reshape(p_re_padded_clu, 344, 397, 2500);

%% extract best pixels (JID bin)

% select period ranges `short`
c_padded_ape = padded_ape(:, b_beg:b_end, :, :);
c_padded_masks = padded_masks(:, b_beg:b_end, :, :);
c_padded_clu = padded_clu(:, b_beg:b_end, :);

full_coeherence = zeros(8, 344, 344);
concat_all_px = cell(1,8);

sub_idx_valid_for_any_scale = zeros(344, 1);

for scale_k = 1:8
    fprintf("Scale %d\n", scale_k)
    
    % select only periods in the defined range for each meta-rhythm
    subset_ape = c_padded_ape(:, valid_scales(scale_k, :) == 1, :, :);
    subset_mask = c_padded_masks(:, valid_scales(scale_k, :) == 1, :, :);
    subset_maksed = subset_ape .* subset_mask;
    subset_clu = c_padded_clu(:, valid_scales(scale_k, :) == 1, :);
    
    % I take all subjetcs
    valid_subset_masked = subset_maksed(:, :, :, :);
    valid_subset_clu = subset_clu(:, :, :);
    n_subs = size(valid_subset_masked, 1);
    n_scales = size(valid_subset_masked, 2);
    re_valid_subset_masked = reshape(valid_subset_masked, n_subs, n_scales, []);
    
    ori_idx = 1:344;
    valid_idx = ori_idx(:);
    valid_for_scale = zeros(1, n_subs);
    best_pixel = zeros(1, n_subs);
    for kk = 1:n_subs
        one_sub = squeeze(valid_subset_clu(kk, :, :));
        one_sub_masked = squeeze(re_valid_subset_masked(kk, :, :));
        clu_idx = unique(one_sub(:));
        if length(clu_idx) == 1
            valid_for_scale(kk) = 0;
        else
            valid_for_scale(kk) = 1;
            no_0_vals = clu_idx(2:end);
            list_of_sums = zeros(1, length(no_0_vals));
            for hj = 1:length(no_0_vals)
                list_of_sums(hj) = sum(one_sub(:) == no_0_vals(hj));
            end
            [~, Iv] = max(list_of_sums);
            clu_idx_max = no_0_vals(Iv);
            one_sub(one_sub ~= clu_idx_max) = 0;
            one_sub(one_sub == clu_idx_max) = 1;
            sum_over_scales = squeeze(sum(one_sub_masked .* one_sub, 1));
            [~, bp] = max(sum_over_scales);
            best_pixel(kk) = bp;
        end
    end
    
    % select only the subject if they have at least some significant pixels
    % in the considered range
    sub_idx_valid_for_scale = valid_idx(valid_for_scale == 1);
    sub_pix_valid_for_scale = best_pixel(valid_for_scale == 1)';
    
    sub_idx_valid_for_any_scale(sub_idx_valid_for_scale) = 1;
    
    for IDXA = 1:length(sub_idx_valid_for_scale)
       
            % subject pixel extraction
            sub_id_a = sub_idx_valid_for_scale(IDXA);
            sub_px = sub_pix_valid_for_scale(IDXA);
            
            jid_idx = find(OriginalIdx == sub_id_a);
            if isempty(jid_idx)
                continue
            end
            jid = JID{jid_idx};
            
            jid = permute(jid, [3,2,1]); % T, 50, 50
            ll = size(jid, 1);
            re_jid = reshape(jid, ll, 2500);
            sel_px_a = re_jid(:, sub_px)';
            
            % pull all best pixels together
            concat_all_px{scale_k} = [concat_all_px{scale_k}, zscore(sel_px_a)];

    end
end
        


%% Bootstrap - Spectral coherence
n_boot = 1000;
tt2 = days(Period(b_beg:b_end));
booted = zeros(8, n_boot);

parfor scale_k = 1:8
    for j = 1:n_boot
        fprintf("Doing %d - %d\n", scale_k, j)
        all_pxs = concat_all_px{scale_k};

        % block boostrap
        pxa = get_boot(all_pxs);
        pxa = pxa(1:6001);
        pxb = get_boot(all_pxs);
        pxb = pxb(1:6001);

        beg_w = tt2(find(valid_scales(scale_k, :)==1, 1, 'first'));
        end_w = tt2(find(valid_scales(scale_k, :)==1, 1, 'last'));

        fb = cwtfilterbank( 'SignalLength',min(length(pxa), length(pxb)),'SamplingPeriod',hours(1), 'VoicesPerOctave',32, 'PeriodLimits', [hours(2) hours(80*24)]);

        % coherence
        [wcoh, ~, period, coi] = wcoherence(pxa, pxb, hours(1), ...
            'VoicesPerOctave', fb.VoicesPerOctave);

        period = days(period);
        coi = days(coi);
        % remove cone-of-influence (coi)
        cw_mask = getcoimaskcwt(wcoh, period, coi);
        wcoh_m = cw_mask .* wcoh;

        % select only the max coherence in the range of frequencies
        % considered
        mean_spec = nanmean(wcoh_m, 2);
        valid_win = (period >= beg_w ) & (period <= end_w);
        subset_mean_spec = mean_spec(valid_win == 1);
        subsub_max_spec = max(subset_mean_spec);

        booted(scale_k, j) = subsub_max_spec;
    end
end



%% block boostrap
function boot = get_boot(x)
ntimes = length(x);
block = 24;
n_win = floor(ntimes / block);
boot = zeros(1, ntimes);
for k = 1:n_win
    v = randi(ntimes - block - 1);
    boot((k - 1) * block + 1:k * block) = x(v+1:v + block);
end
end
