function[F] = NormalNP(data, alpha,F_0)
%NormalNP: Normal graph network propagation.
%   data:   data matrix where rows are samples and columns are features
%   alpha:  tuning parameter of the algorithm
%   F_0: correlation coefficients with label information
N=data;
% m: number of samples, n: number of features
[m, n] = size(N);


% RO: correlation coefficient matrix
W=ones(n,n);
N=N-ones(m,1)*mean(N);
for i=1:n
    for j=i+1:n   
     c=sum(N(:,i).*N(:,j))/(sqrt(sum(N(:,i).^2))*sqrt(sum(N(:,j).^2)));
     W(i,j)=c;
     W(j,i)=W(i,j);
    end
end

Sum_R = sum(abs(W), 2);
Sum_C = sum(abs(W), 1);
S = zeros(n, n);
for i = 1 : n
    for j = 1 : n
            
        S(i, j) = W(i, j) / sqrt(Sum_R(i)) / sqrt(Sum_C(j));
        
    end
end

[U,D,V] = svd(S);
s = max(abs(diag(D)));
D = D/s;
S = U*D*V';

F = zeros(length(alpha),n);

for t=1:length(alpha)
% set initial values

    Y=F_0;
    F(t,:) = F_0;

% do iterations

    for i = 1 : 10000
        F_old = F(t,:);
        F(t,:) = alpha(1,t)*F_old*S + (1 - alpha(1,t))* Y;
        if max(abs(F(t,:) - F_old)) < 1e-8
            break
        end
    end
    if i == 10000
        disp('The algorithm doesn''t converge!')
    end
end
F=F';
