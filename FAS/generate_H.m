function H = generate_H(K,N,L,T,lambda)

H = zeros(N,K);

for k = 1:K

    hk = zeros(N,1);

    theta = pi*rand(L,1);
    phi = pi*rand(L,1);

    pathloss = 1e-8;

alpha = sqrt(pathloss)*...
(randn(L,1)+1j*randn(L,1))/sqrt(2*L);
    for l = 1:L

        rho = [sin(theta(l))*cos(phi(l));
               cos(theta(l))];

        for n = 1:N

            hk(n) = hk(n) + ...
                alpha(l)*exp(-1j*2*pi/lambda * ...
                T(:,n)'*rho);

        end

    end

    H(:,k) = hk;

end

end