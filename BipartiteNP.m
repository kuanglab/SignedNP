function[Y, F] = BipartiteNP(data, label, alpha)
%BipartiteNP Bipartite network propagation.
%   data:   data matrix where each row represent a sample and each column represent a feature
%   label:  label of samples which takes value in {-1,0,1}, 1:case
%   group; -1: control group; 0: test group
%   alpha:  tuning parameter of the algorithm


% normalize the data matrix
Data =abs(data);
[m, n] = size(data);
Sum_R = sum(Data, 2);
Sum_C = sum(Data, 1);
S = zeros(m, n);
for i = 1 : m
    for j = 1 : n
        if data(i, j) ~=0
            S(i, j) = data(i, j) / sqrt(Sum_R(i)) / sqrt(Sum_C(j));
        end
    end
end
[U,D,V] = svd(S);
s =max(abs(diag(D(1:m,1:m))));
D = D/s;
S = U*D*V';

Y_0 = label;
F_0 = zeros(n, 1);

% set initial values
Y = Y_0;
F = F_0;

% do iterations
for i = 1 : 10000
    Y_old = Y;
    F_old = F;
    Y = alpha * S * F_old + (1 - alpha) * Y_0;
    F = alpha * S' * Y_old + (1 - alpha) * F_0;
    if max(abs([Y - Y_old; F - F_old])) < 1e-9
        break
    end
end
if i == 10000
    disp('The algorithm doesn''t converge!')
end

