clc;
clear;
close all;

%% PARAMETERS
K = 4;
N = 4;
L = 4;

fc = 5e9;
c = 3e8;
lambda = c/fc;

P_dBm = -5;
Pmax = 10^((P_dBm-30)/10);

B = 1e6;

N0_dBm = -174;

sigma2_dBm = N0_dBm + 10*log10(B);

sigma2 = 10^((sigma2_dBm-30)/10);

Ifp = 50;
MC =1000;

%% Region sizes
A_norm = 0.5:0.5:6;

sumrate = zeros(size(A_norm));

%% LOOP OVER REGION SIZE
for a_idx = 1:length(A_norm)

    A = A_norm(a_idx)*lambda;

    rate_mc = zeros(MC,1);

    for mc = 1:MC

        %% INITIALIZE POSITIONS
        T = A*rand(2,N);

        

        %% INITIALIZE W
        W = randn(N,K)+1j*randn(N,K);
        W = sqrt(Pmax)*W/norm(W,'fro');
     

       % H = generate_H(K,N,L,T,lambda);

        %% FIX CHANNEL PARAMETERS FOR THIS MC REALIZATION
H = zeros(N,K);

for k = 1:K

    theta = pi*rand(L,1);
    phi = pi*rand(L,1);

    pathloss = 1e-8;

    sigma = sqrt(pathloss)*...
        (randn(L,1)+1j*randn(L,1))/sqrt(2*L);

    rho = zeros(2,L);

    for l = 1:L

        rho(:,l) = ...
            [sin(theta(l))*cos(phi(l));
             cos(theta(l))];

    end

    channel(k).sigma = sigma;
    channel(k).rho = rho;

    %% BUILD INITIAL CHANNEL
    hk = zeros(N,1);

    for l = 1:L

        for n = 1:N

           hk(n) = hk(n) + ...
    sigma(l)*exp(-1j*2*pi/lambda * ...
    (T(:,n)'*rho(:,l)));

        end

    end

    H(:,k) = hk;

end

        %% FP ITERATIONS
        for iter = 1:Ifp

            %% UPDATE lambda AND beta
            lambda_k = zeros(K,1);
            beta_k = zeros(K,1);

            for k = 1:K

                hk = H(:,k);

                signal = abs(hk'*W(:,k))^2;

                interf = 0;

                for j = 1:K
                    if j ~= k
                        interf = interf + abs(hk'*W(:,j))^2;
                    end
                end

                gamma = signal/(interf + sigma2);

                lambda_k(k) = gamma;

                ak = hk'*W(:,k);

                Bk = interf + signal + sigma2;

                beta_k(k) = ak/Bk;

            end

            %% UPDATE W
            C = zeros(N,N);
            D = zeros(N,K);

            for k = 1:K

                hk = H(:,k);

                C = C + ...
                    (1+lambda_k(k))*abs(beta_k(k))^2 ...
                    *(hk*hk');

                D(:,k) = ...
                    (1+lambda_k(k))*conj(beta_k(k))*hk;

            end

            mu = 1e-3;

            W = (C + mu*eye(N))\D;

            W = sqrt(Pmax)*W/norm(W,'fro');

            %% CREATE CHANNEL STRUCTURE FROM CURRENT H
for k = 1:K

    channel(k).h = H(:,k);

    

end

%% EXACT GRADIENT POSITION UPDATE
T = optimize_positions(...
    T,W,channel,...
    lambda_k,beta_k,...
    lambda,A,lambda/2,...
    20,10,1e-3,...
    K,N,L*ones(K,1));

%% UPDATE H USING SAME CHANNEL PARAMETERS
%% UPDATE H USING SAME CHANNEL PARAMETERS
for k = 1:K

    hk = zeros(N,1);

    for l = 1:L

        sigma = channel(k).sigma(l);
        rho = channel(k).rho(:,l);

        for n = 1:N

            hk(n) = hk(n) + ...
                sigma*exp(-1j*2*pi/lambda * ...
                (T(:,n)'*rho));

        end

    end

    H(:,k) = hk;

end
            %% UPDATE CHANNEL AFTER MOVEMENT
           

        end

        %% FINAL SUM RATE
        R = 0;

        for k = 1:K

            hk = H(:,k);

            signal = abs(hk'*W(:,k))^2;

            interf = 0;

            for j = 1:K
                if j ~= k
                    interf = interf + abs(hk'*W(:,j))^2;
                end
            end

            SINR = signal/(interf + sigma2);

            R = R + log2(1 + SINR);

        end

        rate_mc(mc) = R;
if mod(mc,100)==0
    fprintf('MC = %d/%d\n',mc,MC);
end
    end

    sumrate(a_idx) = mean(rate_mc);

    fprintf('A/lambda = %.1f   Rate = %.4f\n',...
        A_norm(a_idx),sumrate(a_idx));

end

%% SMOOTH CURVE
sumrate_smooth = smoothdata(sumrate,'sgolay',5);

%% PLOT
figure;

plot(A_norm,sumrate_smooth,...
    'r-o',...
    'LineWidth',2,...
    'MarkerSize',8);

xlabel('Normalized Region Size A/\lambda');
ylabel('Sum-Rate (bps/Hz)');

title('Normalized Region Size vs Sum-Rate');

grid on;