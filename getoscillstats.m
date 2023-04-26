function [ClusteredOut,clusteredVal,fcluster,DAperiodic]= getoscillstats(realval, bootval);
% function [ClusteredOut,clusteredVal,fcluster,DAperiodic]= getoscillstats(realval, bootval);
% uses the clustering set in LIMO EEG 
% input: 
% realval, in n*n*number_of_scales , based on JID_periodgram_cwt_3d
% or JID_periodgram_cwt_3d_gpu
% Bootval, same dimension as reaval*nboot 
% output: 
% ClusteredOut, boolean locations of significant aperiodic deflections (MCC 0.05)
% clusteredVal, uncorrected cluster numbers bases on p < 0.05
% fcluster, the selected cluster numbers from culsteredVal
% DAperiodic, The spectrogram with the aperiodic component removed using mean bootstrap;
%
% Depends on: 
% find_adj_matrix https://github.com/CODELABLEIDEN/TapDataAnalysis
% LIMO EEG (cluster functions) 


% Set default 
pval = 0.05; % p threshold for clustering 
minchan = 5; % minimum number of neighbouring pixels needed for clustering 
thresh = 100*(1-(pval/2));% bootsrap threshold (considering two-tailed stats) 
nM = find_adj_matrix(50,1); % Neighb. matrix

% reshape
bootval_r = reshape(bootval,2500,size(bootval,3),size(bootval,4));
realval_r = reshape(realval,2500,size(realval,3)); 
% Get thresholded values 
booleanval = realval>prctile(bootval,thresh,4);
booleanval_r = reshape(booleanval,2500,size(booleanval,3));



% run limo clustering (2D) for real data 
[clusteredVal numCluster] = limo_findcluster(booleanval_r,nM,minchan);


% Refer to remove aperiodic component (this is done to make the power vals
% comparable)
ref_aperiodic = mean(bootval_r(:,:,:),3);
realval_r_aperiodic = realval_r-ref_aperiodic;
DAperiodic = reshape(realval_r_aperiodic,50,50,size(realval_r_aperiodic,2)); 
%% Estimate probability of a given pixel being identified according to the clustering
% And the max 
%cluster = parcluster;
%parfor (boot = 1:size(bootval,4), 7);
for boot = 1:size(bootval,4);
%parfor (boot = 1:size(bootval,4),cluster)

    tmp_bootrealval = bootval_r(:,:,boot);
    tmp_bootbootval = bootval_r(:,:,[1:size(bootval,4)]~=boot);
    tmp_D_fromaperiodic = bootval_r(:,:,boot) - ref_aperiodic; 
    
    tmp_booleanval = tmp_bootrealval>prctile(tmp_bootbootval,thresh,3);
    
    [tmp_clusterslabel, nclust] = limo_findcluster(tmp_booleanval,nM,minchan);
    
    %Size of clusters in this boot
    tmp_Dval = []; 
    for n = 1:nclust
    tmp_idx = [tmp_clusterslabel==n];
    tmp_Dval = [tmp_Dval sum(tmp_D_fromaperiodic(tmp_idx))];     
    end
    % Collect the boot values
    if ~isempty(tmp_Dval)
    D_boot(boot) = max(tmp_Dval);      
    else
    D_boot(boot) = 0;    
    end
end



%% now keep only those clusters which are both real and < pval on the boot based on max D values 
for n = 1:numCluster
    idx = [];
    idx = [clusteredVal==n];
    Dval = [sum(realval_r_aperiodic(idx))];   
    num_chance(n,1) = Dval > prctile(D_boot,thresh);
    
end

fcluster = [1:numCluster]; fcluster(~num_chance) = []; % list clusters that survived 
clusteredVal_select = ismember(clusteredVal,fcluster);
ClusteredOut = reshape(clusteredVal_select,50,50,size(clusteredVal_select,2)); 

end