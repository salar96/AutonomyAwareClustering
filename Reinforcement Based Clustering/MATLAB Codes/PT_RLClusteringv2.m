%% This code implements the actual DA algorithm where we increase the 
% number of clusters when the condition for phase transition is satisfied.

%% Constructing Data Set 6
C1 = [0 0]; C2 = [2 0]; C3 = [1 2]; 
C4 = [4 0]; C5 = [6 0]; C6 = [5 2]; 
C7 = [2 3.8]; C8 = [4 3.8]; C9 = [3 5.5];
C10 = [3,2];
rng('default');
Centers = [C1; C2; C3; C4; C5; C6; C7; C8; C9; C10];
Np = 200;
count = 1;
X = zeros(size(Centers,1)*Np, 2);
for i = 1 : size(Centers,1)
    for j = 1 : Np
        x = normrnd(Centers(i,1),0.175);
        y = normrnd(Centers(i,2),0.175);
        X(count,:) = [x y];
        count = count + 1;
    end
end

scatter(X(:,1),X(:,2),'.');
[M, N] = size(X);

%% Setting for DA parameters

K = 1; Tmin = 1e-3; alpha = 0.99; PERTURB = 0.0001; STOP = 1e-2; Kmax = 9;
T = 50; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
rho = Px; Py = 1; D_Arr = zeros(500,1);
K_Arr = zeros(500,1); Beta_Arr = zeros(500,1); K_Arr1 = zeros(500,1);


T_P = zeros(K,K,M);
for j = 1 : K
    for k = 1 : K
        if j ~= k
            T_P(k,j,:) = 1/(K*K);
        end
    end
    T_P(j,j,:) = (K-1)/K;
end
v = VideoWriter('my_simulation_video_With_TP.mp4', 'MPEG-4');  % or 'Motion JPEG AVI'
v.FrameRate = 10;  % frames per second
open(v);
SPLIT = 0.15; inds = []; flag = 0;
while T >= Tmin
    fprintf('%d %d\n', K, T);
    if(flag == 0)
        if(K < Kmax) %phase 1, duplicate codevectors
			YY = [Y;Y] + PERTURB*randn(2*K,N);
            Pyy = 0.5*[Py;Py];
            KK = 2*K;
            T_P = zeros(K,K,M);
            for j = 1 : KK
                for k = 1 : KK
                    if j ~= k
                        T_P(k,j,:) = 2/(KK*KK);
                    end
                end
                T_P(j,j,:) = (KK-2)/KK;
            end
        else %phase 2, do nothing
            YY = Y;
            Pyy = Py;
            KK = K;
            for j = 1 : KK
                for k = 1 : KK
                    if j ~= k
                        T_P(k,j,:) = 1/(KK*KK);
                    end
                end
                T_P(j,j,:) = (KK-1)/KK;
            end
        end
        L_old = inf;
        
        while 1
            [D,D_Act] = distortion(X,YY,M,N,KK,T_P);
            P = association_weights(D, T, KK);
            Y = centroid_location(KK, T_P, M, P, X);
            Pyy = P'*Px;
            %Y = Y + PERTURB*rand(size(Y));
            L = -T*Px'*log(sum(exp(-D/T),2));
            if(norm(L-L_old) < STOP)
                break;
            end
            L_old = L;
        end

        % determine the distinct codevectors
        Y = [];
        Py = [];
        dist=2*SPLIT;
        
        for i = 1:KK
            for j = 1:size(Y,1)
                dist = norm(YY(i,:) - Y(j,:));
                if(dist < SPLIT)
                    break;
                end
            end
            if(dist > SPLIT)
                Y = [Y;YY(i,:)];
                inds = [inds;i];
                Py = [Py;Pyy(i)];
            else
                Py(j) = Py(j) + Pyy(i);
            end
            
            K = size(Y,1);
            if(K > Kmax)
                [sortPy, sortPy_ind] = sort(Py,'descend');
                Y = Y(sortPy_ind(1:Kmax),:);
                Py = sortPy(1:Kmax);
                flag = 1;
                K = Kmax;
            end
        end
    elseif(flag == 1)
        while(1)
            [D,D_Act] = distortion(X,Y,M,N,K,T_P);
            P = association_weights(D, T, K);
            Y = centroid_location(K, T_P, M, P, X);
            Pyy = P'*Px;
            %Y = Y + PERTURB*rand(size(Y));
            L = -T*Px'*log(sum(exp(-D/T),2));
            if(norm(L-L_old) < STOP)
                break;
            end
            L_old = L;
        end
    end
    %D_Arr(count,1) = Dis;
    K_Arr(count,1) = K;
    Beta_Arr(count) = 1/T;
    count = count + 1;
        
    T = T*alpha;
    %scatter(X(:,1),X(:,2),'.'); hold on;
    %scatter(Y(:,1),Y(:,2),'d','filled'); title(T); hold off;
    %frame = getframe(gcf);       % gcf = current figure
    %writeVideo(v, frame);  
end
close(v);
plot(X(:,1),X(:,2),'+b','MarkerSize',16);hold on
plot(Y(:,1),Y(:,2),'*r','MarkerSize',20);hold off
figure;
D_Arr(D_Arr == 0) = [];
K_Arr(K_Arr == 0) = [];
K_Arr1(K_Arr1 == 0) = [];
Beta_Arr(Beta_Arr == 0) = [];
%scatter(K_Arr,D_Arr);
%scatter(K_Arr,Beta_Arr);
plot(Beta_Arr, K_Arr);
