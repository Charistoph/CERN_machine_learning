mat1 = csvread('benchmark_data/output_m3_maxcomp_12.csv',12,0);

baseline_std = mat1(1,:);
baseline_meand = mat1(2,:);
baseline_medid = mat1(3,:);

mat2 = transpose(csvread('ml_output_matlab/results.csv'));

m_mean = mat2(1,:);
m_std = mat2(2,:);

mat3 = transpose(csvread('ml_output_tensorflow/tf_output_benchmarks.csv'));

tf_mean = mat3(1,:);
tf_std = mat3(2,:);

m_diff=m_std-baseline_std;
tf_diff=tf_std-baseline_std;

for i=1:5
  m_diff_perc(i)=m_std(i)/baseline_std(i);
  tf_diff_perc(i)=tf_std(i)/baseline_std(i);
end

m_diff_perc;
tf_diff_perc;

baseline_std
m_std
tf_std
