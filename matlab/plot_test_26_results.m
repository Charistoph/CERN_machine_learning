data1 = csvread('ml_output_matlab/Data_Test_26_Plots/0_500k.csv');
data2 = csvread('ml_output_matlab/Data_Test_26_Plots/500_1000k.csv');
data3 = csvread('ml_output_matlab/Data_Test_26_Plots/1000_1640k.csv');

x = linspace(floor(min(data1(:,2))),round(max(data1(:,2)),-1)+10)
y(1:100) = 1
plot(x,y,data1(:,2),data1(:,1), '.')
title('Test 26: 0 - 500k Events')
set(get(gca, 'XLabel'), 'String', 'Calculation Time');
set(get(gca, 'YLabel'), 'String', 'Parameter / Benchmark');

x = linspace(floor(min(data2(:,2))),round(max(data2(:,2)),-1)+10)
plot(x,y,data2(:,2),data2(:,1), '.')
title('Test 26: 500k - 1000k Events')
set(get(gca, 'XLabel'), 'String', 'Calculation Time');
set(get(gca, 'YLabel'), 'String', 'Parameter / Benchmark');

x = linspace(floor(min(data3(:,2))),round(max(data3(:,2)),-1)+10)
plot(x,y,data3(:,2),data3(:,1), '.')
title('Test 26: 1000k - 1640k Events')
set(get(gca, 'XLabel'), 'String', 'Calculation Time');
set(get(gca, 'YLabel'), 'String', 'Parameter / Benchmark');
