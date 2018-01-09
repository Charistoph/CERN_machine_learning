% DO NOT RUN THIS FILE ALONE
% RUN THROUGH RUNNING *** ml_matlab.m ***

%overVal = abs(res) > 0.5;

%for i=1:size(netout,1)
%  underVal(i,:) = abs(res(i,:)) < 0.5;
%  overVal(i,:) = abs(res(i,:)) > 0.5;
%end
%
%sum(underVal,2)
%sum(overVal,2)
%
%resUnder1 = res(underVal(1,:));
%resOver1 = res(overVal(1,:));
%resUnder2 = res(underVal(2,:));
%resOver2 = res(overVal(2,:));
%resUnder3 = res(underVal(3,:));
%resOver3 = res(overVal(3,:));
%resUnder4 = res(underVal(4,:));
%resOver4 = res(overVal(4,:));
%resUnder5 = res(underVal(5,:));
%resOver5 = res(overVal(5,:));


%for i=1:size(netout,1)
%  figure(i),clf,hist(res(underVal(i,:)),50);
%  results(i,:)=[mean(res(underVal(i,:))),std(res(underVal(i,:)))];
%end

count_res=[0,0,0,0,0];
pos_res2=[];
resUnder2=[];
pos_res1=[];
resUnder1=[];
pos_res3=[];
resUnder3=[];
pos_res4=[];
resUnder4=[];
pos_res5=[];
resUnder5=[];

for i=1:size(res,2)
  for j=1:size(res,1)
    if abs(res(j,i))<0.5
      count_res(j)=count_res(j)+1;
    end
    if abs(res(1,i))<0.5
      pos_res1=[pos_res1,i];
      resUnder1 = [resUnder1,res(1,i)];
    end
    if abs(res(2,i))<0.5
      pos_res2=[pos_res2,i];
      resUnder2 = [resUnder2,res(2,i)];
    end
    if abs(res(3,i))<0.5
      pos_res3=[pos_res3,i];
      resUnder3 = [resUnder3,res(3,i)];
    end
    if abs(res(4,i))<0.5
      pos_res4=[pos_res4,i];
      resUnder4 = [resUnder4,res(4,i)];
    end
    if abs(res(5,i))<0.5
      pos_res5=[pos_res5,i];
      resUnder5 = [resUnder5,res(5,i)];
    end
  end
end
count_res

figure(1),clf,hist(resUnder1,50);
figure(2),clf,hist(resUnder2,50);
figure(3),clf,hist(resUnder3,50);
figure(4),clf,hist(resUnder4,50);
figure(5),clf,hist(resUnder5,50);

for i=1:5
  saveas(figure(i),['ml_analyze/figure_' num2str(i) '.fig']);
end
