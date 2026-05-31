function [lambda_vec,beta_vec,sumRate] = ...
    update_lambda_beta(channel,W,sigma2,K)

lambda_vec = zeros(K,1);
beta_vec = zeros(K,1);

sumRate = 0;

for k = 1:K

    hk = channel(k).h;

    signal = abs(hk'*W(:,k))^2;

    interf = 0;

    for j = 1:K
        if j ~= k
            interf = interf + abs(hk'*W(:,j))^2;
        end
    end

    gamma = signal/(interf + sigma2);

    lambda_vec(k) = gamma;

    ak = hk'*W(:,k);

    Bk = sigma2;

    for j = 1:K
        Bk = Bk + abs(hk'*W(:,j))^2;
    end

    beta_vec(k) = ak/Bk;

    sumRate = sumRate + log2(1+gamma);

end

end