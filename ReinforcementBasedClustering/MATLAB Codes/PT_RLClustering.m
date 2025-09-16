%% This code implements the DA algorithm with p(k|j,i) transition prob.
idx = 1;
[X,K,T_P,M,N] = data_RLClustering(idx); close all;
X_org = X;
[X, mu, sigma] = zscore(X);

%% Setting for DA parameters

Tmin = 0.001; alpha = 0.99; PERTURB = 0.0001; STOP = 1e-2;
T = 12; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
rho = Px;

beta_cr = 0;
%v = VideoWriter('my_simulation_video_With_TP.mp4', 'Motion JPEG 2000');  % or 'Motion JPEG AVI'
%v.FrameRate = 10;  % frames per second
%open(v);
while T >= Tmin
    L_old = inf;
    while 1
        [D,D_Act] = distortion_RLClustering(X,Y,M,N,K,T_P);
        num = exp(-D/T);
        den = repmat(sum(num,2),[1 K]);
        P = num./den;
        Py = P'*Px;
        for l = 1:K
            T_slice = squeeze(T_P(l, :, :))';   
            W = (1/M) * P .* T_slice;           
            row_weights = sum(W, 2);            
            numerator = row_weights' * X;
            denominator = sum(row_weights);
            Y(l, :) = numerator / denominator;
        end
        Y = Y + PERTURB*rand(size(Y));
        if isnan(Y)
            pp = 1;
        end
        L = -T*Px'*log(sum(exp(-D/T),2));
        if(norm(L-L_old) < STOP)
            break;
        end
        L_old = L;
    end
    %T_cr = critical_beta(X, Y, K, M, T_P, P, rho);
    %T_cr = critical_beta_NewDelta(X, Y, K, M, T_P, P, rho);
    %fprintf('%d %d \n',T,T_cr);
    disp(T);
    T = T*alpha;
    %scatter(X(:,1),X(:,2),'.'); hold on;
    %scatter(Y(:,1),Y(:,2),'d','filled'); title(T); hold off;
    %frame = getframe(gcf);       % gcf = current figure
    %writeVideo(v, frame);  
end
%close(v);
Y = Y.*sigma + mu; X = X.*sigma + mu;

idx = cell(M,1);
[~,idx_O] = max(P,[],2);
%col = [0.545, 0.000, 0.000;
%    0.000, 0.392, 0.000;
%    0.000, 0.000, 0.545;
%    1.000, 0.549, 0.000];
%symbols = {'$\mathbf{y_1}$', '$\mathbf{y_2}$', '$\mathbf{y_3}$', '$\mathbf{y_4}$'};
%Y1 = [Y(1,1)-0.25, Y(1,2) - 0.25;
%    Y(2,1)+0.25, Y(2,2)-0.25;
%    Y(3,1)+0.25, Y(3,2)-0.25;
%    Y(4,1)-0.25, Y(4,2)-0.25];
%col1 = zeros(K,3);
%col2 = zeros(K,3);
% for j = 1 : K
% 
%    idx{j} = find(idx_O==j);
%    col1(j,:) = 0.2*col(j,:) + 0.8*[1,1,1];
%    col2(j,:) = 0.5*col(j,:) + 0.5*[1,1,1];
%    scatter(X(idx{j},1),X(idx{j},2),750,'filled','MarkerEdgeColor',col2(j,:),...
%            'MarkerFaceColor',col1(j,:),'LineWidth',0.01); hold on;
%    scatter(Y(j,1),Y(j,2),2000,'p','MarkerEdgeColor', 'k', 'MarkerFaceColor', col(j,:),...
%        'LineWidth',1);
%    text(Y1(j,1), Y1(j,2), symbols{j}, 'Interpreter','latex', ...
%         'FontSize',80, 'HorizontalAlignment','center', ...
%         'VerticalAlignment','bottom'); % adjust alignment as needed
% end
% hold off; xlim([-7 7]); ylim([-6 6]); axis square; box on; 
% set(gca, 'FontSize', 80); set(gca, 'LineWidth', 1.0);
% xticks([-5 0 5]); yticks([-5 0 5]);

%savefig('Setup_2D.fig');
%print(gcf, 'Setup_2D.png', '-dpng', '-r600');
cluster_labels = zeros(length(X),1);
for i = 1 : length(X)
    cluster_labels(i) = find(P(i,:)>=0.90);
end

[coeff, score, ~] = pca(X); % X is your N x 7 data matrix
gscatter(score(:,1), score(:,2), cluster_labels)
xlabel('PC 1'); ylabel('PC 2');
title('PCA-based Cluster Visualization');

%parallelcoords(X, 'Group', cluster_labels, 'Standardize', 'on');
%title('Parallel Coordinates Plot for 7D Data');
% hold on;
% for i = 1 :size(X23_org,1)
%     if i <= 62516
%         plot(1:7, X23_org(i,:), 'Color','g');
%         if i == 62516
%             hold off; figure;
%         end
%     else
%         if i == 62517
%             hold on;
%         end
%          plot(1:7, X23_org(i,:), 'Color','b');
%     end
%     disp(i);
% end
% hold off;

Y_z = zscore(Y);
X_z = zscore(X);
sm = min([Y_z; X_z],[],'all');
Y_z = Y_z - sm; X_z = X_z - sm;

col = zeros(size(Y,1),3);
col(1,:) = [0.10 0.25 0.55];   % dark blue
col(2,:) = [0.55 0.10 0.15];   % dark red / maroon
col(3,:) = [0.10 0.45 0.20];
Y_z = [Y_z Y_z(:,1)];
X_z = [X_z X_z(:,1)];
for i = 1 : size(Y,1)
    %col(i,:) = rand([1,3]);
    polarplot(Y_z(i,:), 'Color',col(i,:)); hold on;
end

% for i = 1 : size(X,1)
%     if cluster_labels(i) == 1
%         polarplot(X_z(i,:),'--','Color',col(cluster_labels(i),:));
%         continue;
%     end
%     if cluster_labels(i) == 2
%         polarplot(X_z(i,:),'--','Color',col(cluster_labels(i),:));
%         continue;
%     end
%     if cluster_labels(i) == 3
%         polarplot(X_z(i,:),'--','Color',col(cluster_labels(i),:));
%         continue;
%     end
% end
hold off;