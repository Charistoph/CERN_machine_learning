batch = 'results_ex1_5_neurons';
foldername = 'ml_analyze/results';
temp=load(strcat(foldername, '/', batch, '.csv'));

l=length(temp)
ntr=0;

while ntr<l
    ntr=ntr+1;
    para1(ntr)=temp(ntr,1);
    para2(ntr)=temp(ntr,2);
    para3(ntr)=temp(ntr,3);
end

% plot histogramms
f=figure;
subplot(3,1,1);
hist(para2,100)

subplot(3,1,2);
hist(para2,100)

subplot(3,1,3);
hist(para3,100)
saveas(f, strcat(foldername, '/', batch, '_histo.png'));

% statistics
stat1 = sum(para1<1)/ntr;
stat2 = sum(para2<1)/ntr;
stat3 = sum(para3<1)/ntr;

ml_train_result  = strcat( ...
{'para1: '}, num2str(stat1), ...
{', para2: '}, num2str(stat2), ...
{', para3: '}, num2str(stat3) ...
)

dlmwrite(strcat(foldername, '/', batch, '_stats.csv'),ml_train_result,'delimiter','','-append');
