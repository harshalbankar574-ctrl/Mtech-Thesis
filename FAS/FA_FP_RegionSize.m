% =========================================================================
% FA (FP-Based): Sum-Rate vs. Normalised Region Size  A/lambda
% Reproduces FA(FP) curve -- Fig. 5 of Cheng et al., IEEE Commun. Lett. 2024
% =========================================================================
clear; clc; close all;
rng(42);

%% --- Parameters (Section IV of paper) -----------------------------------
par.N      = 4;                          % transmit FAs
par.K      = 4;                          % single-antenna UTs
par.Lk     = 4;                          % paths per UT
par.D      = 0.5;                        % min spacing [wavelengths]
par.lam    = 1;                          % normalised wavelength
par.f0     = 5e9;                        % carrier freq [Hz]
par.B      = 1e6;                        % bandwidth [Hz]
par.N0_dBm = -174;                       % noise PSD [dBm/Hz]
par.cell_r = 1500;                       % cell radius [m]
par.p_dBm  = 5;
par.p      = 10^(par.p_dBm/10)*1e-3;    % power budget [W]

par.Ifp    = 50;    % Algorithm 2 outer iterations
par.I      = 20;    % Algorithm 1 gradient iterations
par.u_ini  = 1;     % initial step size (smaller = more stable)
par.u_min  = 1e-4;  % min step size
par.N_mc   = 200;   % Monte Carlo runs (1000 for paper quality)

A_vec = 1:0.5:6;
nA    = length(A_vec);
R_FA  = zeros(1,nA);

fprintf('FA(FP) Sum-Rate vs A/lambda  [N_mc=%d]\n\n', par.N_mc);

%% --- Simulation ---------------------------------------------------------
for ia = 1:nA
    par.A = A_vec(ia);
    acc   = 0;
    for mc = 1:par.N_mc
        [ch, sig2] = gen_channel(par);
        [W0, T0]   = init_random(par);
        [W,  T ]   = alg2_FP(W0, T0, ch, sig2, par);
        acc        = acc + sum_rate(W, T, ch, sig2, par);
    end
    R_FA(ia) = acc / par.N_mc;
    fprintf('  A/lam=%.1f  R=%.4f bps/Hz\n', A_vec(ia), R_FA(ia));
end

%% --- Plot ---------------------------------------------------------------
figure('Color','w','Position',[200 150 620 450]);
plot(A_vec, R_FA, '-o', 'Color',[0.00 0.45 0.74], ...
     'LineWidth',2, 'MarkerSize',8, 'MarkerFaceColor',[0.00 0.45 0.74]);
grid on; box on;
xlabel('Normalised Region Size  A/\lambda','FontSize',13);
ylabel('Sum-Rate [bps/Hz]','FontSize',13);
title('FA (FP-Based): Sum-Rate vs. Normalised Region Size  (p = -5 dBm)','FontSize',12);
legend('FA (FP)','Location','northwest','FontSize',12);
xlim([1 6]);  set(gca,'FontSize',12,'XTick',1:6);
saveas(gcf,'FA_FP_SumRate_vs_RegionSize.png');
fprintf('\nSaved: FA_FP_SumRate_vs_RegionSize.png\n');


%% ========================================================================
%%  FUNCTIONS
%% ========================================================================

% -------------------------------------------------------------------------
% Channel generation
%   sigma_{k,l} ~ CN(0, mu_k/Lk),   mu_k = linear path loss
%   rho_{k,l}   = [sin(theta)cos(phi); cos(theta)]   (2-D, Eq.2)
%   sig2_k      = N0*B  (noise power, same for all k -- paper does NOT
%                        divide by mu; SINR already carries path loss via h_k)
% -------------------------------------------------------------------------
function [ch, sig2] = gen_channel(par)
    K  = par.K;   Lk = par.Lk;

    r_ut  = par.cell_r * sqrt(rand(K,1));          % uniform in disk
    PL_dB = 92.5 + 20*log10(par.f0/1e9) ...
                 + 20*log10(max(r_ut,1e-3)/1e3);
    mu    = 10.^(-PL_dB/10);                        % linear path loss

    ch.Lk    = Lk;
    ch.sigma = cell(K,1);
    ch.rho   = cell(K,1);

    for k = 1:K
        % CN(0, mu_k/Lk) per path
        ch.sigma{k} = sqrt(mu(k)/(2*Lk)) * (randn(Lk,1)+1j*randn(Lk,1));
        theta = pi*rand(1,Lk);
        phi   = pi*rand(1,Lk);
        ch.rho{k} = [sin(theta).*cos(phi); cos(theta)];  % 2 x Lk
    end

    % Noise power: sigma^2_k = N0*B  (W)
    sig2 = 10^(par.N0_dBm/10)*1e-3 * par.B * ones(K,1);
end

% -------------------------------------------------------------------------
% Scalar channel  h_k(t)  -- Eq.(2)
% -------------------------------------------------------------------------
function v = hk(t, k, ch, par)
    v = 0;
    for l = 1:ch.Lk
        v = v + ch.sigma{k}(l) * ...
            exp(-1j*(2*pi/par.lam)*(t'*ch.rho{k}(:,l)));
    end
end

% -------------------------------------------------------------------------
% Full channel matrix  H  (N x K)
% -------------------------------------------------------------------------
function H = H_mat(T, ch, par)
    H = zeros(par.N, par.K);
    for n = 1:par.N
        for k = 1:par.K
            H(n,k) = hk(T(:,n), k, ch, par);
        end
    end
end

% -------------------------------------------------------------------------
% Sum-rate  R = sum_k log2(1+SINR_k)  -- Eq.(4)
% -------------------------------------------------------------------------
function R = sum_rate(W, T, ch, sig2, par)
    H = H_mat(T, ch, par);
    R = 0;
    for k = 1:par.K
        sk   = abs(H(:,k)'*W(:,k))^2;
        tot  = sum(abs(H(:,k)'*W).^2);
        R    = R + log2(1 + sk/(tot - sk + sig2(k)));
    end
end

% -------------------------------------------------------------------------
% Feasibility: t inside [0,A]^2 AND >= D from all other antennas
% -------------------------------------------------------------------------
function ok = feasible(t, n, T, par)
    ok = all(t >= 0) && all(t <= par.A);
    if ~ok, return; end
    for np = 1:par.N
        if np~=n && norm(t-T(:,np)) < par.D
            ok = false; return;
        end
    end
end

% -------------------------------------------------------------------------
% Initialise: random feasible T, MRT-scaled W
% -------------------------------------------------------------------------
function [W0, T0] = init_random(par)
    T0 = zeros(2,par.N);
    for n = 1:par.N
        placed = false;
        for attempt = 1:20000
            tc = par.A * rand(2,1);
            ok = true;
            for np = 1:n-1
                if norm(tc-T0(:,np)) < par.D, ok=false; break; end
            end
            if ok, T0(:,n)=tc; placed=true; break; end
        end
        if ~placed   % fallback: place on grid
            T0(:,n) = [mod(n-1,2)*par.D*2; floor((n-1)/2)*par.D*2];
        end
    end
    W0 = (randn(par.N,par.K)+1j*randn(par.N,par.K))/sqrt(2);
    pwr = real(trace(W0*W0'));
    if pwr > 0, W0 = W0*sqrt(par.p/pwr); end
end

% -------------------------------------------------------------------------
% Lemma 1: update lambda and beta
% -------------------------------------------------------------------------
function [lam, bet] = upd_lam_bet(W, H, sig2, par)
    lam = zeros(par.K,1);  bet = zeros(par.K,1);
    for k = 1:par.K
        ak  = W(:,k)'*H(:,k);
        Bk  = sig2(k) + sum(abs(H(:,k)'*W).^2);
        ink = Bk - abs(ak)^2;
        lam(k) = abs(ak)^2/(ink + sig2(k));
        bet(k) = ak/Bk;
    end
end

% -------------------------------------------------------------------------
% Optimise W -- Eq.(8), bisection on regulariser (Eq.9-10)
% -------------------------------------------------------------------------
function W = opt_W(lam, bet, H, sig2, par)
    N=par.N; K=par.K; p=par.p;
    C = 1e-10*eye(N);  D = zeros(N,K);
    for k = 1:K
        C = C + (1+lam(k))*abs(bet(k))^2*(H(:,k)*H(:,k)');
        D(:,k) = (1+lam(k))*conj(bet(k))*H(:,k);
    end
    C = (C+C')/2;

    W_unc = C\D;
    if real(trace(W_unc*W_unc')) <= p
        W = W_unc; return;
    end

    % Bisection
    [U,Lv] = eig(C);
    ev      = max(real(diag(Lv)),0);
    UDDtUH  = U'*D*D'*U;
    pfn     = @(mu) sum(diag(real(UDDtUH.*conj(UDDtUH)))./(ev+mu).^2) - p;

    mu_hi = 1e-8;
    for i=1:80, if pfn(mu_hi)<0, break; end; mu_hi=mu_hi*10; end
    mu_lo = 0;
    for i=1:100
        mu_m=(mu_lo+mu_hi)/2;
        if pfn(mu_m)>0, mu_lo=mu_m; else, mu_hi=mu_m; end
        if mu_hi-mu_lo<1e-15, break; end
    end
    W = (C+(mu_lo+mu_hi)/2*eye(N))\D;
end

% -------------------------------------------------------------------------
% f_n(t_n) -- Eq.(11)
% -------------------------------------------------------------------------
function val = fn_val(n, tn, T, W, lam, bet, ch, par)
    WWH = W*W';  val = 0;
    for k = 1:par.K
        hkn  = hk(tn, k, ch, par);
        dkn  = (1+lam(k))*abs(bet(k))^2*real(WWH(n,n));
        snp  = 0;
        for np=1:par.N
            if np~=n, snp=snp+WWH(n,np)*hk(T(:,np),k,ch,par); end
        end
        ckn  = (1+lam(k))*(bet(k)*W(n,k) - abs(bet(k))^2*snp);
        val  = val + 2*real(conj(hkn)*ckn) - dkn*abs(hkn)^2;
    end
end

% -------------------------------------------------------------------------
% Gradient of f_n -- Eq.(12)
% Correct derivation:
%   d/dt Re{conj(h_k(t)) * c} = Re{ d/dt h_k(t)^* * c }
%   dh_k(t)/dt_n = sum_l sigma_{k,l} * (-j2pi/lam) * rho_{k,l}
%                  * exp(-j2pi/lam * t^T rho_{k,l})
% so gradient of 2Re{h_k^* c_{kn}} w.r.t. t is:
%   sum_l 2Re{ conj(sigma_{k,l}*(−j2pi/lam)*exp(-j..)) * c_{kn} } * rho
% = sum_l (4pi/lam)|tau| sin(phase + angle(tau)) * rho  [matches Eq.12]
%
% gradient of -d_{kn}|h_k(t)|^2:
%   = -d_{kn} * sum_{l,l'!=l} |sigma_l sigma_l'|*(4pi/lam)/d_{kn}
%               * sin(2pi/lam * t^T rho_{ll'} + theta_{ll'}) * rho_{ll'}
% -------------------------------------------------------------------------
function g = grad_fn(n, tn, T, W, lam, bet, ch, par)
    WWH = W*W';  g = zeros(2,1);  lv = par.lam;
    for k = 1:par.K
        sk = ch.sigma{k};  rk = ch.rho{k};  Lk = ch.Lk;
        snp = 0;
        for np=1:par.N
            if np~=n, snp=snp+WWH(n,np)*hk(T(:,np),k,ch,par); end
        end
        dkn = (1+lam(k))*abs(bet(k))^2*real(WWH(n,n));
        ckn = (1+lam(k))*(bet(k)*W(n,k) - abs(bet(k))^2*snp);

        % Term 1: gradient of 2*Re{h_k(tn)^* * c_{kn}}
        % h_k(t)^* = sum_l sigma_l^* exp(+j2pi/lam t^T rho_l)
        % d/dt_n [h_k^* * c] contributes:
        %   sum_l conj(sigma_l)*(j2pi/lam)*exp(j2pi/lam t^T rho_l) * c_{kn}
        % 2*Re{...} w.r.t. t_n:
        for l = 1:Lk
            tau   = conj(sk(l))*ckn;   % tau_n^{k,l} = sigma^*_{k,l} * c_{kn}
            phase = (2*pi/lv)*(tn'*rk(:,l)) + angle(tau);
            g = g + (4*pi/lv)*abs(tau)*sin(phase)*rk(:,l);
        end

        % Term 2: gradient of -d_{kn}*|h_k(tn)|^2
        % |h_k|^2 = sum_{l,l'} sigma_l * sigma_l'^*
        %           * exp(-j2pi/lam * t^T (rho_l - rho_l'))
        % d/dt_n |h_k|^2 = sum_{l!=l'} |sigma_l sigma_l'|
        %                  * (4pi/lam) * sin(2pi/lam t^T rho_{ll'} + theta_{ll'})
        %                  * rho_{ll'}    (after taking real part)
        % so gradient of -d_{kn}|h_k|^2 carries a (-) sign:
        if dkn > 1e-15
            for l = 1:Lk
                for lp = 1:Lk
                    if lp~=l
                        rho_ll  = rk(:,l)-rk(:,lp);
                        th_ll   = angle(sk(lp))-angle(sk(l));
                        coeff   = abs(sk(l)*sk(lp))*(4*pi/lv)*dkn;
                        phi_ll  = (2*pi/lv)*(tn'*rho_ll)+th_ll;
                        % Note: paper writes coeff/(lam/4pi*d_{kn}) = coeff*4pi/(lam*dkn)
                        % The -d_{kn} * grad|h|^2 => multiply by dkn then negate
                        g = g - coeff*sin(phi_ll)*rho_ll;
                    end
                end
            end
        end
    end
end

% -------------------------------------------------------------------------
% Algorithm 1: gradient ascent + backtracking on each t_n
% -------------------------------------------------------------------------
function T = alg1(T, W, lam, bet, ch, par)
    for it = 1:par.I
        for n = 1:par.N
            tn    = T(:,n);
            g     = grad_fn(n, tn, T, W, lam, bet, ch, par);
            f_cur = fn_val(n, tn, T, W, lam, bet, ch, par);
            u     = par.u_ini;
            while u >= par.u_min
                tn_new = tn + u*g;
                if feasible(tn_new, n, T, par)
                    if fn_val(n, tn_new, T, W, lam, bet, ch, par) > f_cur
                        T(:,n) = tn_new;
                        break;
                    end
                end
                u = u/2;
            end
        end
    end
end

% -------------------------------------------------------------------------
% Algorithm 2: FP outer loop
% -------------------------------------------------------------------------
function [W, T] = alg2_FP(W, T, ch, sig2, par)
    for it = 1:par.Ifp
        H          = H_mat(T, ch, par);
        [lam, bet] = upd_lam_bet(W, H, sig2, par);
        W          = opt_W(lam, bet, H, sig2, par);
        T          = alg1(T, W, lam, bet, ch, par);
    end
end