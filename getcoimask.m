function [cw_mask] = getcoimask(cw, t, coi); 
% [cw_nan] = getcoimaskcwt (cw, t, coi); 
% cw_nan, contains NaN values where coi should be masked; 
% use outputs from function cwt as inputs [cw,t, coi] = cwt(x); 
% Arko Ghosh, Leiden University 
% Enea Ceolini, Leiden University
cw_mask = nan(size(cw));
for i = 1:size(cw,2)

    t_val = min(find(coi(i)<t, 1, 'first'));
    if ~isempty (t_val)
        idx(i) = min(find(coi(i)<t, 1, 'first'))-1;
    else
            idx(i) = size(cw,1);
    end
    cw_mask(:, i) = 1;
    cw_mask(idx(i):end, i) = deal(NaN);

end