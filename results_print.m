% DO NOT RUN THIS FILE ALONE
% RUN THROUGH RUNNING *** ml_matlab.m ***

% print histos & save outputs

results=zeros(size(netout,1),2);
res=targets-netout;
for i=1:size(netout,1)
  figure(i),clf,hist(res(i,:),50);
  results(i,:)=[mean(res(i,:)),std(res(i,:))];
end

dt = datestr(now,'yyyy.mm.dd-HH:MM:SS')
path = strcat(pwd,'/ml_output_matlab/',dt)
resultspath = strcat(path,'/results.csv')
benchmark_resultspath = strcat(path,'/benchmark_results.csv')
mkdir(path)

for i=1:5
saveas(figure(i),[path '/figure_' num2str(i) '.fig']);
end

csvwrite('ml_output_matlab/results.csv',results)
csvwrite(resultspath,results)
