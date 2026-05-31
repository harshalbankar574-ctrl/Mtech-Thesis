function W = update_beamforming(...
    channel,Wold,lambda_vec,beta_vec,...
    Pmax,K,N)

C = zeros(N,N);
Dmat = zeros(N,K);

for k = 1:K

    hk = channel(k).h;

    C = C + ...
        (1+lambda_vec(k))*abs(beta_vec(k))^2 ...
        *(hk*hk');

    Dmat(:,k) = ...
        (1+lambda_vec(k))*conj(beta_vec(k))*hk;

end

%% Bisection for mu
mu_low = 0;
mu_high = 1000;

for iter = 1:50

    mu = (mu_low + mu_high)/2;

    Wtemp = (C + mu*eye(N))\Dmat;

    power_now = real(trace(Wtemp*Wtemp'));

    if power_now > Pmax
        mu_low = mu;
    else
        mu_high = mu;
    end

end

W = (C + mu*eye(N))\Dmat;

end