clc;
clear;
close all;

%% Parameters
K = 4;              % Users
N = 4;              % Fluid antennas
Lk = 4*ones(K,1);   % Paths

fc = 5e9;
c = 3e8;
lambda_c = c/fc;

A = 2*lambda_c;
D = lambda_c/2;

Pmax_dBm = 5;
Pmax = 10^((Pmax_dBm-30)/10);

noise_dBm = -174 + 10*log10(1e6);
sigma2 = 10^((noise_dBm-30)/10);

Ifp = 20;
I_gd = 20;

u_ini = 10;
u_min = 1e-3;

%% Initialize antenna positions
T = A*rand(2,N);

%% Generate channels
[channel] = generate_channel(K,N,Lk,T,lambda_c);

%% Initialize Beamforming
W = (randn(N,K)+1j*randn(N,K))/sqrt(2);
W = sqrt(Pmax)*W/norm(W,'fro');

sumrate_store = zeros(Ifp,1);

%% FP Iterations
for iter = 1:Ifp

    %% Update lambda and beta
    [lambda_vec,beta_vec,sumRate] = update_lambda_beta(...
        channel,W,sigma2,K);

    sumrate_store(iter) = sumRate;

    %% Update W using Eq. (8)
    W = update_beamforming(channel,W,...
        lambda_vec,beta_vec,Pmax,K,N);

    %% Optimize antenna positions
    T = optimize_positions(...
        T,W,channel,...
        lambda_vec,beta_vec,...
        lambda_c,A,D,...
        I_gd,u_ini,u_min,K,N,Lk);

    %% Update deterministic channel using moved positions
for k = 1:K

    hk = zeros(N,1);

    for l = 1:Lk(k)

        rho = channel(k).rho(:,l);
        sigma = channel(k).sigma(l);

        for n = 1:N

            hk(n) = hk(n) + ...
                sigma*exp(-1j*2*pi/lambda_c * ...
                (T(:,n).'*rho));

        end
    end

    channel(k).h = hk;

end

    fprintf('Iteration %d SumRate = %.4f\n',iter,sumRate);

end

%% Plot
figure;
plot(sumrate_store,'LineWidth',2);
xlabel('Iteration');
ylabel('Sum Rate (bps/Hz)');
grid on;
title('FP-Based Fluid Antenna Optimization');