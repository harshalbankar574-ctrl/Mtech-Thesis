function grad = compute_gradient(...
    n,T,W,channel,...
    lambda_vec,beta_vec,...
    lambda_c,K,Lk)

grad = zeros(2,1);

for k = 1:K

    hk = channel(k).h;
    sigma_k = channel(k).sigma;
    rho_k = channel(k).rho;

    wk = W(:,k);

    ck = (1+lambda_vec(k))*...
        beta_vec(k)*wk(n);

    for l = 1:Lk(k)

        phase = 2*pi/lambda_c * ...
            (T(:,n).'*rho_k(:,l));

        tau = conj(sigma_k(l))*ck;

        grad = grad ...
            - abs(tau)*(pi/lambda_c)*...
            sin(phase + angle(tau))*...
            rho_k(:,l);

    end
end

end