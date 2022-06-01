function [cw_nan] = getcoimaskcwt (cw, t, coi); 
% [cw_nan] = getcoimaskcwt (cw, t, coi); 
% cw_nan, contains NaN values where coi should be masked; 
% use outputs from function cwt as inputs [cw,t, coi] = cwt(x); 
% Arko Ghosh, Leiden University 

cw_nan = nan(size(cw,1),size(cw,2));
for i = 1:size(cw,2)
   try 
   idx(i) = min(find(coi(i)<t));
   catch 
   if abs(round(coi(i)) - max(round(coi))) < 5
   idx(i) = size(cw,1); 
   else
   idx(i) = 1;
   end
   end
   cw_nan(:,i) = abs(cw(:,i));
   cw_nan(idx(i):end,i) = deal(NaN);  
   
end