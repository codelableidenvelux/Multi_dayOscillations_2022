function simil = amariMaxCorr(A)
% A version of Amari correlation

% this is gonna be 1 on the diagonal and I don't want to cosnider it.
for i = 1:size(A,1)
    A(i, i) = -1;
end

% because of possible permutations in the D I need to make sure I find the
% best possible permutation of components that gives the best correlation.
maxCol = max(A,[],1);
simil = mean(maxCol);

