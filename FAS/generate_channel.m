function channel = generate_channel(K,N,Lk,T,lambda_c)

for k = 1:K

    L = Lk(k);

    theta = pi*rand(L,1);
    phi = pi*rand(L,1);

    sigma = (randn(L,1)+1j*randn(L,1))/sqrt(2*L);

    hk = zeros(N,1);

    rho = zeros(2,L);

    for l = 1:L

        rho(:,l) = [sin(theta(l))*cos(phi(l));
                    cos(theta(l))];

        for n = 1:N

            hk(n) = hk(n) + ...
                sigma(l)*exp(-1j*2*pi/lambda_c * ...
                (T(:,n).'*rho(:,l)));

        end
    end

    channel(k).h = hk;
    channel(k).sigma = sigma;
    channel(k).rho = rho;
end

end