%% This code implements the Parameterized MEP.
% No learning here.

%% Setting Hyperparameters
resource_count = 10;
betamin = 1e-6;
betamax = 1e7;
beta_growth_rate = 1.1;
PERTURB = 1e-5;
gamma = 1.0; % discount factor
count_zeta_max = 200;
count_V_max = 200;
tol_V = 0.0001;
count_G_max = 200;
alpha = 0.0010; % gradient learning rate for parameters
%% Train loop

load('clusters_data.mat');
para = data;
para(resource_count+1,:) = [4 7];


action_count = resource_count + 1; % number of possible actions
state_count = length(para);  % total number of states
mu = (1/action_count)*ones(state_count, action_count);
Q = zeros(state_count, action_count);
V = zeros(state_count, 1);

beta = betamin;
while beta < betamax
    disp('beta=');
    disp(beta);
    beta = beta * beta_growth_rate;
    count_zeta = 1;
    F_old = inf;
    while count_zeta <= count_zeta_max
        count1 = 1;
        [P, C] = prob_costMatrixv3(para, state_count, action_count);
        TT = log(P);
        TT(isinf(TT)) = 0;
        AvC1 = squeeze(sum(P.*C,1));
        AvC2 = (gamma/beta)*squeeze(sum(P.*TT,1));
        
        AvC = AvC1 + AvC2;
        V_old = V;
        Q = zeros(state_count, action_count);
        V = zeros(state_count, 1);
        while count1 <= count_V_max
            Q = (AvC + gamma*squeeze(sum(P.*repmat(V,[1,state_count, action_count]),1)));
            tmp1 = repmat(min(Q,[],2), [1 action_count]);
            tmp2 = sum(exp((-(beta/gamma))*(Q-tmp1)),2);
            V = -(gamma/beta)*((-(beta/gamma))*min(Q,[],2) + log(tmp2));
            if (norm(V-V_old)/norm(V) < 0.001)
                break;
            else
                V_old = V;
            end
            count1 = count1 + 1;
        end
        
        if norm(F_old-V)/norm(V) < tol_V
            break;
        else
            F_old = V;
        end
        
        num = exp(-(beta/gamma)*(Q - tmp1));
        den = repmat(tmp2, [1, action_count]);
        mu = num./den;
        
        % Computing the derivatives
        
        count2 = 1;
        G1_old = zeros(state_count, resource_count);
        G2_old = zeros(state_count, resource_count);
        while count2 < count_G_max
            
            G1 = zeros(state_count, resource_count);
            G2 = zeros(state_count, resource_count);
            C1 = zeros(state_count, state_count);
            C2 = zeros(state_count, state_count);
            
            for j = 1 : resource_count
                
                if count2 > 1
                    if abs(sum(G1_old(action_count+1:end,j))) < 0.5 && abs(sum(G2_old(action_count+1:end,j))) < 0.5
                        continue;
                    end
                end
                
                mu_reshape = reshape(mu, 1, state_count, action_count);
                mu_rep = repmat(mu_reshape, [state_count 1 1]);
                THETA = sum(mu_rep.*P,3);
                THETA = THETA';
                % Precompute the differences
                diff1 = 2 * (para(j, 1) - para(:, 1));
                diff2 = 2 * (para(j, 2) - para(:, 2));
                
                % Update C1 and C2 in a vectorized manner
                C1(j, :) = diff1;
                C2(j, :) = diff2;
                C1(:, j) = diff1;
                C2(:, j) = diff2;
                
                %Zero out specific regions of C1 and C2

                C1(1:resource_count+1, 1:end) = 0;
                C1(resource_count+2:end, resource_count+1:end) = 0;
                C2(1:resource_count+1, 1:end) = 0;
                C2(resource_count+2:end, resource_count+1:end) = 0;
          
                % C1(logical(eye(size(C1)))) = 0;
                % C1(1:action_count, 1:action_count) = 0;
                % C1(action_count+1:end, :) = 0;
                % C1(action_count, 1:action_count) = 0; % f and d to d
                % C1(action_count, action_count+1:end) = 0; % node to d
                % C1(1:action_count-1, action_count) = 0; % d to f
                % C2(logical(eye(size(C2)))) = 0;
                % C2(1:action_count, 1:action_count) = 0;
                % C2(action_count+1:end, :) = 0;
                % C2(action_count, 1:action_count) = 0; % f and d to d
                % C2(action_count, action_count+1:end) = 0; % node to d
                % C2(1:action_count-1, action_count) = 0; % d to f


                C1 = THETA.*C1;
                C2 = THETA.*C2;
                
                G1_old1 = G1(:,j);
                G2_old2 = G2(:,j);
                count3 = 1;
                while count3 < 500
                    
                    G1(:,j) = sum(C1,2) + gamma*THETA*G1(:,j);
                    G2(:,j) = sum(C2,2) + gamma*THETA*G2(:,j);
                    
                    if norm(G1_old1 - G1(:,j))/norm(G1(:,j))<0.001 && norm(G2_old2 - G2(:,j))/norm(G2(:,j)) < 0.001
                        break;
                    else
                        G1_old1 = G1(:,j);
                        G2_old2 = G2(:,j);
                        count3 = count3 + 1;
                    end
                end 
                
                tmp2 = sum(G1(action_count+1:end,j));
                tmp3 = sum(G2(action_count+1:end,j));

                
                para(j,1) = para(j,1) - alpha*tmp2;
                
                para(j,2) = para(j,2) - alpha*tmp3;

                G1_old(:,j) = G1(:,j);
                G2_old(:,j) = G2(:,j);
            end
            count2 = count2 + 1;
            
            if all(abs(sum(G1_old(action_count+1:end, :))) < 0.5) && ...
               all(abs(sum(G2_old(action_count+1:end, :))) < 0.5)
                break;
            end

        end
        count_zeta = count_zeta + 1;
    end
    para(1:resource_count,:) = para(1:resource_count,:) + PERTURB*randn(resource_count,2);
    % disp('Small Cell Location');
    % disp(para(1:resource_count,:));
    
end
[P, C] = prob_costMatrixv3(para, state_count, action_count);
TT = log(P);
TT(isinf(TT)) = 0;
AvC = squeeze(sum(P.*C,1));
D = policy_eval(gamma,P,AvC, state_count, action_count);
plot(para(action_count+1:end,1), para(action_count+1:end,2), '.');
hold on; axis square
plot(para(action_count,1), para(action_count,2),'.');
plot(para(1:resource_count,1), para(1:resource_count,2),'*',MarkerSize=12);
hold off
%legend('Users','Base station','Small Cells');
