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

result(1,:)=baseline_std;
result(2,:)=matlab_std;
result(3,:)=tensorflow_std;

csvwrite('benchmark_data/benchmark_result.csv',result)
