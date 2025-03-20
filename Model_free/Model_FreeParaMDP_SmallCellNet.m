%% Reinforcement Learning for parameter identification in Small Cell Network Design Problem
MV=0.8;
COUNT_ZETA = 50;
delta_zeta = 0.0001; %for gradient estimation
resource_count = 6;
para = zeros(6,2);
para(6,:) = [4,7];
Centroids = [1.5 2.5; 3 1; 5 1; 8 1; 10, 2.5; 9, 5];
rng('default');
rng(1);
for i = 1 : length(Centroids)
    for j = 1 : 5+randi([0,5],1,1)
        R = 0.1 + 0.5*rand(1);
        theta = 2*pi*rand(1);
        randx = Centroids(i,1) + R*cos(theta);
        randy = Centroids(i,2) + R*sin(theta);
        para = [para; randx, randy];
    end
end
action_count = resource_count + 1; % number of possible actions
state_count = length(para); % total number of states
tot_epi = 1000; betamin = 0.001; betamax = 1000; 
choice1 = [1, 2]; mu = (1/action_count)*ones(state_count,action_count); V = zeros(state_count);
beta_arr = 10.^(linspace(log10(betamin),log10(betamax),tot_epi));

for ep = 1 : tot_epi
    beta = beta_arr(ep);
    count_zeta = 1;
    while count_zeta <= COUNT_ZETA
        count_zeta = count_zeta + 1;
        itT = 0;
        Q = zeros(state_count,action_count);
        
        %%% Learning the policy %%%
        
        while itT < 1000
            itT = itT + 1;
            Q_old = Q;
            state = ShortestPath_reset(action_count, state_count);
            itr = 0;
            epsilon = 1;
            count = ones(state_count,action_count);
            while itr < 200
                pmf = [epsilon, 1 - epsilon];
                sample_size = 1;
                ep_choice = randsample(choice1,sample_size,true,pmf);
                if ep_choice == 2
                    [~, action] = max(mu(state,:));
                else
                    action = randi(action_count);
                end
                
                count(state, action) = count(state, action) + 1;
                [next_state, cost, done] = ShortestPath_stoch2(state, action, para, resource_count, action_count, state_count);
                Q(state, action) = Q(state, action) + (count(state, action))^(-MV)*(cost + V(next_state)...
                    - Q(state, action));
                V(state) = (-1/beta)*(-beta*min(Q(state,:)) + log(sum(exp(-beta*(Q(state,:) - min(Q(state,:)))))));
                
                Q_D = min(Q(state,:));
                ct = 0;
                for i = 1 : action_count
                    if isinf(sum(exp(-beta*(Q(state,:) - Q_D))))
                        if isinf(exp(-beta*(Q(state,i) - Q_D)))
                            ct = ct + 1;
                            mu(state,:) = 0;
                            mu(state,i) = 1;
                            if ct >= 2 && i == action_count
                                for j = 1 : action_count
                                    if isinf(exp(-beta*(Q(state,j) - Q_D)))
                                        mu(state,j) = 1/ct;
                                    end
                                end
                            end
                            break;
                        else
                            mu(state,i) = exp(-beta*(Q(state,i) - Q_D))/sum(exp(-beta*(Q(state,:) - Q_D)));
                        end
                    else
                        mu(state,i) = exp(-beta*(Q(state,i) - Q_D))/sum(exp(-beta*(Q(state,:) - Q_D)));
                    end
                end
                itr = itr + 1;
                epsilon = epsilon * 0.95;
                state = next_state;
                if done
                    break;
                end
            end
        end
        
        %%% Learning the derivatives %%%
        
        G1 = zeros(state_count,action_count);
        G2 = zeros(state_count,action_count);
        K1 = cell(resource_count,1); 
        K2 = cell(resource_count,1);
        for t = 1 : resource_count
            K1{t} = zeros(state_count,action_count); 
            K2{t} = zeros(state_count,action_count);
        end

        for j = 1 : resource_count

            para_p = para;
            para_p(j,:) = para_p(j,:) + [delta_zeta, 0];
            itT = 0;
            while itT < 500
                G1_old = G1;
                itT = itT + 1;
                state = ShortestPath_reset(action_count, state_count);
                state_p = ShortestPath_reset(action_count, state_count);
                if state_p ~= state
                    state_p = state;
                end
                itr = 0;
                count = ones(state_count,action_count);
                while itr < 50
                    itr = itr + 1;
                    [~, action] = max(mu(state,:));
                    
                    count(state, action) = count(state, action) + 1;
                    [next_state, cost, done] = ShortestPath_stoch2(state, action, para, resource_count, action_count, state_count);
                    [next_state_p, cost_p, ~] = ShortestPath_stoch2(state_p, action, para_p, resource_count, action_count, state_count);
                    while next_state_p ~= next_state
                        [next_state, cost, done] = ShortestPath_stoch2(state, action, para, resource_count, action_count, state_count);
                        [next_state_p, cost_p, ~] = ShortestPath_stoch2(state_p, action, para_p, resource_count, action_count, state_count);
                    end
                    if state ~= j && action ~= j && next_state ~= j
                        K1{j}(state, action) = K1{j}(state, action) + (1/count(state,action))^(MV)*(0 + G1(next_state, j) - K1{j}(state, action));
                        G1(state, j) = mu(state,:) * K1{j}(state,:)';
                    else
                        tmp = (cost_p - cost)/delta_zeta;
                        K1{j}(state, action) = K1{j}(state, action) + (1/count(state, action))^(MV)*(tmp + G1(next_state, j) - K1{j}(state, action));
                        G1(state, j) = mu(state,:) * K1{j}(state,:)';
                    end
                    if done
                        break;
                    end
                    state = next_state;
                    state_p = next_state_p;
                end
            end
            itT = 0;
            para_p = para;
            para_p(j,:) = para_p(j,:) + [0, delta_zeta];
            while itT < 500
                itT = itT + 1;
                state = ShortestPath_reset(action_count, state_count);
                state_p = ShortestPath_reset(action_count, state_count);
                if state_p ~= state
                    state_p = state;
                end
                itr = 0;
                count = ones(state_count,action_count);
                while itr < 50
                    itr = itr + 1;
                    action = randi(action_count);
                    count(state, action) = count(state, action) + 1;
                    [next_state, cost, done] = ShortestPath_stoch2(state, action, para, resource_count, action_count, state_count);
                    [next_state_p, cost_p, ~] = ShortestPath_stoch2(state_p, action, para_p, resource_count, action_count, state_count);
                    while next_state_p ~= next_state
                        [next_state, cost, done] = ShortestPath_stoch2(state, action, para, resource_count, action_count, state_count);
                        [next_state_p, cost_p, ~] = ShortestPath_stoch2(state_p, action, para_p, resource_count, action_count, state_count);
                    end
                    if state ~= j && action ~= j && next_state ~= j
                        K2{j}(state, action) = K2{j}(state, action) + (1/count(state,action))^(MV)*(0 + G2(next_state, j) - K2{j}(state, action));
                        G2(state, j) = mu(state,:) * K2{j}(state,:)';
                    else
                        tmp = (cost_p - cost)/delta_zeta;
                        K2{j}(state, action) = K2{j}(state, action) + (1/count(state, action))^(MV)*(tmp + G2(next_state, j) - K2{j}(state, action));
                        G2(state, j) = mu(state,:) * K2{j}(state,:)';
                    end
                    if done
                        break;
                    end
                    state = next_state;
                    state_p = next_state_p;
                end
            end
        end

        %%% Updating the parameters (small cell locations) via gradient
        %%% descent
        for j = 1 : resource_count
            tmp2 = 0; tmp3 = 0;
            for state = 1 : state_count   
                tmp2 = tmp2 + G1(state,j);
                tmp3 = tmp3 + G2(state,j);
            end
            if norm(G1(:,j)) > 80
                para(j,1) = para(j,1) - 0.01/(norm(G1(:,j)))*tmp2;
            else
                para(j,1) = para(j,1) - 0.002*tmp2;
            end
            if norm(G2(:,j)) > 80
                para(j,2) = para(j,2) - 0.01/norm(G2(:,j))*tmp3;
            else
                para(j,2) = para(j,2) - 0.002*tmp3;
            end
        end
    end
    disp('beta = ');
    disp(beta);
    disp(para(1:resource_count,:));
end
%%
plot(para(action_count+1:end,1), para(action_count+1:end,2), '.');
hold on; axis square
plot(para(action_count,1), para(action_count,2),'.');
plot(para(1:resource_count,1), para(1:resource_count,2),'*','MarkerSize', 12);
hold off
%legend('Users','Base station','Small Cells');

% function s = ShortestPath_reset(action_count, state_count)
%     s = randi(state_count);
%     while s == action_count
%         s = randi(state_count);
%     end
% end
%%
mu_n = mu(resource_count+2:end,:);
% Define colormap for clusters
cmap = hsv(size(mu_n, 2)); % Generate a distinct color for each cluster

% Plot nodes belonging to each cluster with unique colors
hold on; axis square;
for cluster_idx = 1:size(mu_n, 2)-1
    % Find indices of nodes belonging to the current cluster
    cluster_nodes = find(mu_n(:, cluster_idx));

    % Plot the nodes of the current cluster
    plot(para(cluster_nodes+resource_count+1, 1), para(cluster_nodes+resource_count+1, 2), 'o', 'Color', cmap(cluster_idx, :));
    
    plot(para(cluster_idx, 1), para(cluster_idx, 2), '*', 'MarkerSize', 12, 'Color', 'black');
end