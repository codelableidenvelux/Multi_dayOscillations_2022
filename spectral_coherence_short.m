% Spectral Coherence for `short` Periods
% - For each subject and each scale find the best JID bin in the biggest
% cluster
% - for each pair of people (thus JID bins) we operate the spectral coherence
%
% Enea Ceolini, Leiden University

%% load data
load('./data/JIDcompiledHourly_sel.mat')
load('./data/ape_padded_and_non_padded_v5.mat')
load('./data/ForCluster.mat', 'Period')
load('./data/to_plot_fig3_short_v5.mat')
load('./data/to_plot_fig4_short_v5.mat')
load('./data/perferred_ranks_short_v5.mat')

%% initialization
b_beg = 190;
b_end = 336;
n_subs = length(preferred_ranks);

% this is due to python loading a MATLAB tensor of size X, Y, Z in the 
% order Z, Y, X
re_padded_clu = reshape(padded_clu, n_subs, 397, 50, 50);
p_re_padded_clu = permute(re_padded_clu, [1,2,4,3]);
padded_clu = reshape(p_re_padded_clu, n_subs, 397, 2500);


%% doing the spectral coherence

% select period ranges `short`
c_padded_ape = padded_ape(:, b_beg:b_end, :, :);
c_padded_masks = padded_masks(:, b_beg:b_end, :, :);
c_padded_clu = padded_clu(:, b_beg:b_end, :);

full_coeherence = zeros(5, n_subs, n_subs);

sub_idx_valid_for_any_scale = zeros(n_subs, 1);

for scale_k = 1:5
    fprintf("Scale %d\n", scale_k)
    
    % select only periods in the defined range for each meta-rhythm
    subset_ape = c_padded_ape(:, valid_scales(scale_k, :) == 1, :, :);
    subset_mask = c_padded_masks(:, valid_scales(scale_k, :) == 1, :, :);
    subset_maksed = subset_ape .* subset_mask;
    subset_clu = c_padded_clu(:, valid_scales(scale_k, :) == 1, :);
    
    % I select all the subjects 
    valid_subset_masked = subset_maksed(:, :, :, :);
    valid_subset_clu = subset_clu(:, :, :);
    n_subs = size(valid_subset_masked, 1);
    n_scales = size(valid_subset_masked, 2);
    re_valid_subset_masked = reshape(valid_subset_masked, n_subs, n_scales, []);
    
    % selection of best pixel (JID bin) in the largest cluster
    ori_idx = sub_id;
    valid_idx = 1:n_subs;
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
    ori_idx_valid_for_scale = ori_idx(valid_for_scale == 1);
    sub_pix_valid_for_scale = best_pixel(valid_for_scale == 1)';
    
    sub_idx_valid_for_any_scale(sub_idx_valid_for_scale) = 1;
    
    for IDXA = 1:length(sub_idx_valid_for_scale)
        for IDXB = IDXA + 1:length(sub_idx_valid_for_scale)
            
            % subject a - pixel extraction
            sub_id_a = sub_idx_valid_for_scale(IDXA);
            ori_id_a = ori_idx_valid_for_scale(IDXA);
            sub_px = sub_pix_valid_for_scale(IDXA);
            
            jid_idx = find(OriginalIdx == ori_id_a);
            if isempty(jid_idx)
                continue
            end
            jid = JID{jid_idx};
            
            jid = permute(jid, [3,2,1]); % T, 50, 50
            ll = size(jid, 1);
            re_jid = reshape(jid, ll, 2500);
            sel_px_a = re_jid(:, sub_px);
            
            % subject b - pixel extraction
            sub_id_b = sub_idx_valid_for_scale(IDXB);
            ori_id_b = ori_idx_valid_for_scale(IDXB);
            sub_px = sub_pix_valid_for_scale(IDXB);
            
            jid_idx = find(OriginalIdx == ori_id_b);
            if isempty(jid_idx)
                continue
            end
            jid = JID{jid_idx};
            fprintf("\tCoherence %d - %d\n", ori_id_a, ori_id_b);
            
            jid = permute(jid, [3,2,1]); % T, 50, 50
            ll = size(jid, 1);
            re_jid = reshape(jid, ll, 2500);
            sel_px_b = re_jid(:, sub_px);
            
            tt2 = days(Period(b_beg:b_end));
            
            beg_w = tt2(find(valid_scales(scale_k, :)==1, 1, 'first'));
            end_w = tt2(find(valid_scales(scale_k, :)==1, 1, 'last'));
            
            fb = cwtfilterbank( 'SignalLength',min(length(sel_px_b), length(sel_px_a)),'SamplingPeriod',hours(1), 'VoicesPerOctave',32, 'PeriodLimits', [hours(2) hours(80*24)]);
            
            % coherence
            [wcoh, ~, period, coi] = wcoherence(sel_px_a, sel_px_b, hours(1), ...
                'VoicesPerOctave', fb.VoicesPerOctave);
            
            % remove cone of influence
            period = days(period);
            coi = days(coi);
            cw_mask = getcoimask(wcoh, period, coi);
            wcoh_m = cw_mask .* wcoh;
            
            % select only the max coherence in the range of frequencies
            % considered
            mean_spec = nanmean(wcoh_m, 2);
            valid_win = (period >= beg_w ) & (period <= end_w);
            subset_mean_spec = mean_spec(valid_win == 1);
            subsub_max_spec = max(subset_mean_spec);
            
            full_coeherence(scale_k, sub_id_a, sub_id_b) = subsub_max_spec;
            
        end
    end
end

%% 24 hours

a = mean(padded_ape, [1, 3, 4]);
valid_24 = a > prctile(a, 97);

full_coeherence_24 = zeros(n_subs, n_subs);

% subset again to valid scales per scale range k
subset_ape = padded_ape(:, valid_24 == 1, :, :);
subset_mask = padded_masks(:, valid_24 == 1, :, :);
subset_maksed = subset_ape .* subset_mask;
subset_clu = padded_clu(:, valid_24 == 1, :);

valid_subset_masked = subset_maksed(:, :, :, :);
valid_subset_clu = subset_clu(:, :, :);
n_subs = size(valid_subset_masked, 1);
n_scales = size(valid_subset_masked, 2);
re_valid_subset_masked = reshape(valid_subset_masked, n_subs, n_scales, []);

ori_idx = sub_id;
valid_idx = 1:n_subs;
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

sub_idx_valid_for_scale = valid_idx(valid_for_scale == 1);
ori_idx_valid_for_scale = ori_idx(valid_for_scale == 1);
sub_pix_valid_for_scale = best_pixel(valid_for_scale == 1)';

for IDXA = 1:length(sub_idx_valid_for_scale)
    for IDXB = IDXA + 1:length(sub_idx_valid_for_scale)
        
        % a
        sub_id_a = sub_idx_valid_for_scale(IDXA);
        ori_id_a = ori_idx_valid_for_scale(IDXA);
        sub_px = sub_pix_valid_for_scale(IDXA);
        
        jid_idx = find(OriginalIdx == ori_id_a);
        if isempty(jid_idx)
            continue
        end
        jid = JID{jid_idx};
        
        jid = permute(jid, [3,2,1]); % T, 50, 50
        ll = size(jid, 1);
        re_jid = reshape(jid, ll, 2500);
        sel_px_a = re_jid(:, sub_px);
        
        
        sub_id_b = sub_idx_valid_for_scale(IDXB);
        ori_id_b = ori_idx_valid_for_scale(IDXB);
        sub_px = sub_pix_valid_for_scale(IDXB);
        
        jid_idx = find(OriginalIdx == ori_id_b);
        if isempty(jid_idx)
            continue
        end
        jid = JID{jid_idx};
        fprintf("\tCoherence %d - %d\n", ori_id_a, ori_id_b);
        
        jid = permute(jid, [3,2,1]); % T, 50, 50
        ll = size(jid, 1);
        re_jid = reshape(jid, ll, 2500);
        sel_px_b = re_jid(:, sub_px);
        
        
        tt2 = days(Period);
        
        beg_w = tt2(find(valid_24 == 1, 1, 'first'));
        end_w = tt2(find(valid_24 == 1, 1, 'last'));
        
        fb = cwtfilterbank( 'SignalLength',min(length(sel_px_b), length(sel_px_a)),'SamplingPeriod',hours(1), 'VoicesPerOctave',32, 'PeriodLimits', [hours(2) hours(80*24)]);
        
        
        [wcoh, ~, period, coi] = wcoherence(sel_px_a, sel_px_b, hours(1), ...
            'VoicesPerOctave', fb.VoicesPerOctave);
        
        
        period = days(period);
        coi = days(coi);
        cw_mask = getcoimask(wcoh, period, coi);
        wcoh_m = cw_mask .* wcoh;

        mean_spec = nanmean(wcoh_m, 2);

        valid_win = (period >= beg_w ) & (period <= end_w);
        subset_mean_spec = mean_spec(valid_win == 1);
        subsub_max_spec = max(subset_mean_spec);
        
        full_coeherence_24(sub_id_a, sub_id_b) = subsub_max_spec;
        
    end
end