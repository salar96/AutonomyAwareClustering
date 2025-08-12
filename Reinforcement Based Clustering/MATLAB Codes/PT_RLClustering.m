%% This code implements the DA algorithm with p(k|j,i) transition prob.
idx = 4;
[X,K,T_P,M,N] = data_RLClustering(idx); close all;
X_org = X;
[X, mu, sigma] = zscore(X);

%% Setting for DA parameters

Tmin = 0.0005; alpha = 0.99; PERTURB = 0.0001; STOP = 1e-2;
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
%Y = Y.*sigma + mu; X = X.*sigma + mu;

% scatter(X(:,1),X(:,2),90,'filled','MarkerEdgeColor',[0 0.5 0.5],...
%            'MarkerFaceColor',[0 0.7 0.7],'LineWidth',1.5); 
% xlim([-7 7]); ylim([-6 6]); axis square; box on; 
% set(gca, 'FontSize', 25); set(gca, 'LineWidth', 1.0);
% xticks([-5 0 5]); yticks([-5 0 5]);
% hold on; scatter(Y(:,1),Y(:,2),500,'p','MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');hold off;
%savefig('Setup_2D.fig');
%print(gcf, 'Setup_2D.png', '-dpng', '-r600');
% cluster_labels = zeros(length(X),1);
% for i = 1 : length(X)
%     cluster_labels(i) = find(P(i,:)>=0.95);
% end
% 
% [coeff, score, ~] = pca(X); % X is your N x 7 data matrix
% gscatter(score(:,1), score(:,2), cluster_labels)
% xlabel('PC 1'); ylabel('PC 2');
% title('PCA-based Cluster Visualization');
% 
% parallelcoords(X, 'Group', cluster_labels, 'Standardize', 'on');
% title('Parallel Coordinates Plot for 7D Data');
