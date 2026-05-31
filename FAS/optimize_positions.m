function T = optimize_positions(...
    T,W,channel,...
    lambda_vec,beta_vec,...
    lambda_c,A,D,...
    I_gd,u_ini,u_min,K,N,Lk)

for gd_iter = 1:I_gd

    for n = 1:N

        grad = compute_gradient(...
            n,T,W,channel,...
            lambda_vec,beta_vec,...
            lambda_c,K,Lk);

        u = u_ini;

        while u > u_min

            Tnew = T;
            Tnew(:,n) = T(:,n) + u*grad;

            %% Boundary constraint
            Tnew(:,n) = max(Tnew(:,n),0);
            Tnew(:,n) = min(Tnew(:,n),A);

            %% Distance constraint
            feasible = 1;

            for m = 1:N

                if m ~= n

                    dist = norm(Tnew(:,n)-Tnew(:,m));

                    if dist < D
                        feasible = 0;
                    end

                end
            end

            if feasible
                T(:,n) = Tnew(:,n);
                break;
            end

            u = u/2;

        end

    end

end

end