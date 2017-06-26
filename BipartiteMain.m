clc
clear all

% tuning parameter alpha
alpha = [0.95; 0.5; 0.1];

load SampleData

% randomly choose 30% of the samples as test data for prediction.
n = round(length(Label)*0.3);

% estimate the prediction errors
errRate = zeros(length(alpha),1);
L = Label;
IX = randperm(length(Label));
IX = IX(1:n);
L(IX) = 0;

[nn mm]= size(Data);

% coefficients F
F = zeros(nn,length(alpha));

% predict labels
Y = zeros(mm,length(alpha));

for i = 1:length(alpha)
    [Y(:,i), F(:,i)] = BipartiteNP(Data', L', alpha(i));
    Y(Y(:,i)>0,i) = 1;
    Y(Y(:,i)<0,i) = -1;
    errRate(i) = length(find(abs((Label(IX)-Y(IX)))>0))/length(IX);   
end





