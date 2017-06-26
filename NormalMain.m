
clc
clear all
load SampleData

n=size(Data,1);

% correlation coefficients with label information
cc=zeros(1,n);

% tuning parameter alpha
alpha=[0.95,0.5,0.1];

for j=1:n
      D=corrcoef(Data(j,:),Label);
      cc(1,j)=D(1,2);
end

% Learned coefficients F
F = NormalNP(Data',alpha, cc);

