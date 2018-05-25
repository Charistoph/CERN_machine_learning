load('test28.mat');
resT = transpose(res);

% size should be same as inputs and tracks
size(resT)

cov(resT)
trace(cov(resT))