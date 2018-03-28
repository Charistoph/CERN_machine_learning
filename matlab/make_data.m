% DO NOT RUN THIS FILE ALONE
% RUN THROUGH RUNNING *** ml_matlab.m ***

% create inputs and targets of length ntr, remove rows = 0

inputs=zeros(ntr,72);
targets=zeros(ntr,5);

for itr=1:ntr
  if track(itr).ncomp==12
    for i=1:track(itr).ncomp
      for j=1:size(track(1).comp(i).par,1)
      % inputs
        inputs(itr,i*6-5+j)=track(itr).comp(i).par(j);
      end
      inputs(itr,i*6-5)=track(itr).comp(i).weight;
    end
  end
  % targets
  for k=1:size(track(1).tp,1)
    targets(itr,k)=track(itr).tp(k);
  end
end

% How many rows are = 0
count=0
pos=[];
for i=1:size(inputs,1)
  if inputs(i,3)==0
    count=count+1;
    i;
    pos=[pos,i];
  end
end

% Remove rows = 0
count2=0
for i=1:size(inputs,1)
  inputs(i-count2,:)=inputs(i,:);
  targets(i-count2,:)=targets(i,:);
  for j=1:size(pos,2)
    if pos(j)==i
      count2=count2+1;
    end
  end
end
% Cut away empty rows
inputs=inputs(1:ntr-count2,:);
targets=targets(1:ntr-count2,:);

%for i=1:size(targets,1)
%  figure(i),clf,hist(targets(i,:),50)
%end

% remove targets with radius over 0.001
overCounter=0
pos=[];
for i=1:size(targets,1)
  if targets(i,4)*targets(i,4)+targets(i,5)*targets(i,5)>0.001
    overCounter=overCounter+1;
    pos=[pos,i];
  end
end
overCounter

size(inputs)
size(targets)

if datacheck
  count_inputs=0;
  pos_inputs=[];
  count_targets=0;
  pos_targets=[];
  for i=1:size(inputs,1)
    for j=1:size(inputs,2)
      if inputs(i,j)==0
        count_inputs=count_inputs+1;
        pos_inputs=[pos_inputs,j];
      end
    end
    for j=1:size(targets,2)
      if targets(i,j)==0
        count_targets=count_targets+1;
        pos_targets=[pos_targets,j];
      end
    end
  end
  count_inputs
  count_targets
end

% save data_root/matlab_inputs_tagets inputs targets ntr
% try
%   save ml_input/ml_inputs_targets inputs targets
% catch
%   mkdir ml_input
%   save ml_input/ml_inputs_targets inputs targets
% end

% try
%   save /Users/christoph/Documents/coding/CERN_input_data/ml_inputs_targets_ex1_180208_144044_20180307 inputs targets
% catch
%   mkdir ml_input
%   save /Users/christoph/Documents/coding/CERN_input_data/ml_inputs_targets_ex1_180208_144044_20180307 inputs targets
% end
