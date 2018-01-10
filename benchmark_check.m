% DO NOT RUN THIS FILE ALONE
% RUN THROUGH RUNNING *** ml_matlab.m ***

% compares MAD, Matlab & Tensorflow results

mat1 = csvread('benchmark_data/output_m3_maxcomp_12.csv',12,0);

baseline_std = mat1(1,:);
baseline_meand = mat1(2,:);
baseline_medid = mat1(3,:);

mat2 = transpose(csvread('ml_output_matlab/results.csv'));

m_mean = mat2(1,:);
matlab_std = mat2(2,:);

mat3 = transpose(csvread('ml_output_tensorflow/tf_output_benchmarks.csv'));

tf_mean = mat3(1,:);
tensorflow_std = mat3(2,:);

m_diff=matlab_std-baseline_std;
tf_diff=tensorflow_std-baseline_std;

for i=1:5
  m_diff_perc(i)=matlab_std(i)/baseline_std(i);
  tf_diff_perc(i)=tensorflow_std(i)/baseline_std(i);
end

m_diff_perc;
tf_diff_perc;

baseline_std
matlab_std
tensorflow_std

benchmark_result(1,:)=baseline_std;
benchmark_result(2,:)=matlab_std;
benchmark_result(3,:)=tensorflow_std;

for i=1:5
  matlab_worse_than_baseline(i) = benchmark_result(2,i)/benchmark_result(1,i);
end

matlab_worse_than_baseline

csvwrite('benchmark_data/benchmark_result.csv',benchmark_result)

csvwrite(benchmark_resultspath,benchmark_result);

ml_train_log  = strcat(dt, {':     '});
for i=1:5
  ml_train_log = strcat(ml_train_log, num2str(matlab_worse_than_baseline(i)), {'    '});
end
ml_train_log  = strcat(ml_train_log, {'  - '}, num2str(neurons), {' LM, MSE, sgdm, Batchs.,64 '}, num2str(trainMethod))

dlmwrite('ml_output_matlab/ml_train_log.csv',ml_train_log,'delimiter','','-append');