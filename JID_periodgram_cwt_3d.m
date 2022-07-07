function [oJID, times] = JID_periodgram_cwt_3d(JIDs, varargin)
% Calcuates the periodgram for a sequence of JIDs
%    oJID = JID_periogram(JIDs);
%    Positional parameters:
%
%      JIDs           sequence of JIDs sampled at 1 hour. Tensor shape is
%                     expected to be pixels-by-pixels-by-times
%    Optional arguments
%
%
%      permute        functionality to do bootstrapping. If true the pixels
%                     are shuffled independently over the time dimension.
%
%      block          Number of timepoints in  a block. if 1 each time
%                     series is permuted, if > 1 block of that size will
%                     be randonly picked from the original times series
%                     to create the surrogate one.
%
%      limits         Range  of the cwt periodogram for example and default
%                      is [hours(2) hours(80*24)]
%
%      useGPU         true if you want to use gpuArray
%
%   Returns:
%      X            Periodgram of the JID. in the shape
%                   nscales-by-pixels-by-pixels
%      times        Periods in Matlab time format
%
%
%
%
%
% Enea Ceolini, Leiden University, 04/06/2021
% Arko Ghosh, Leiden University, 13/12/2021

p = inputParser;
addRequired(p,'JIDs');
addOptional(p,'permute', false);
addOptional(p,'block', 1);
addOptional(p,'limits', [hours(2) hours(80*24)]);
addOptional(p,'useGPU', false);

parse(p, JIDs, varargin{:});

permute = p.Results.permute;

block = p.Results.block;

useGPU = p.Results.useGPU;

[~, npx, ntimes] = size(JIDs);  % 3D -> (npx, npx, ntimes)
vJIDs = reshape(JIDs, npx * npx, ntimes);  % 2D -> (npx^2, ntimes)

if permute
    if block == 1
        [~, pidx] = sort(rand(npx * npx, ntimes),2); % only permutes the second dimension;
        vJIDs = vJIDs(pidx);
    else
        n_win = floor(ntimes / block);
        boot = zeros(npx * npx, ntimes);
        for j = 1:npx * npx  % for each pixel
            for k = 1:n_win
                v = randi(ntimes - block - 1);
                boot(j, (k - 1) * block + 1:k * block) = vJIDs(j, v+1:v + block);
            end
        end
        vJIDs = boot;
    end
end

px_z = zscore(vJIDs, 0, 2);
fb = cwtfilterbank( 'SignalLength',(size(single(px_z),2)),'SamplingPeriod',hours(1), 'VoicesPerOctave',40,  'PeriodLimits', [hours(2) hours(80*24)]);

x = px_z(1,:);

if useGPU
    x = gpuArray(single(x));
end
[cw,t,coi] = cwt(x,'filterbank', fb);
t = gather(t);
cw = gather(cw);
coi = gather(coi);
cw_mask = getcoimask(cw, t, coi);
cw_x = zeros(2500, length(t));
cw_x(1,:) = sqrt(mean(abs(cw .* cw_mask), 2, 'omitnan'));

for i = 2:size(px_z,1)
    if i == 12
        fprintf("BUG\n")
    end
    x = px_z(i,:);
    if useGPU
        x = gpuArray(single(x));
    end
    cw = cwt(x, 'filterbank', fb);
    cw = gather(cw);
    cw_x(i,:) = sqrt(mean(abs(cw .* cw_mask), 2, 'omitnan'));
    clc; fprintf('Completed cwt on : %d bins of 2500 .\n', i);
    % clear x cw_nan cw coi;
end

oJID = reshape(cw_x,npx,npx,length(t));

times = t;

end
