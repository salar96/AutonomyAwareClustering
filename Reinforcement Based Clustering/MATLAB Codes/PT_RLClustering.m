%% This code implements the actual DA algorithm where we increase the 
% number of clusters when the condition for phase transition is satisfied.

%% Constructing Data Set 1
C1 = [0 0]; C2 = [1 0]; C3 = [0.5 0.9]; 
C4 = [5 0]; C5 = [6 0]; C6 = [5.5 0.9]; 
C7 = [2.5 4.2]; C8 = [3.5 4.2]; C9 = [3 5];
rng('default');
Centers = [C1; C2; C3; C4; C5; C6; C7; C8; C9];
Np = 100;
count = 1;
X = zeros(size(Centers,1)*Np, 2);
for i = 1 : size(Centers,1)
    for j = 1 : Np
        x = normrnd(Centers(i,1),0.125);
        y = normrnd(Centers(i,2),0.125);
        X(count,:) = [x y];
        count = count + 1;
    end
end
%% Constructing Data Set 2
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
X1 = X;
%% Constructing Data Set 3
C11 = [-8 -4]; C21 = [4 -4]; C31 = [4 4]; C41 = [-8 4];
C12 = C11 + [3.5 0]; C13 = C11 + [0 3.5]; C14 = C11 + [3.5 3.5];
C22 = C21 + [3.5 0]; C23 = C21 + [0 3.5]; C24 = C21 + [3.5 3.5];
C32 = C31 + [3.5 0]; C33 = C31 + [0 3.5]; C34 = C31 + [3.5 3.5];
C42 = C41 + [3.5 0]; C43 = C41 + [0 3.5]; C44 = C41 + [3.5 3.5];


rng('default');
Centers = [C11; C12; C13; C14; C21; C22; C23; C24;...
            C31; C32; C33; C34; C41; C42; C43; C44];
Np = 200;
count = 1;
X = zeros(size(Centers,1)*Np, 2);
C = zeros(size(Centers,1)*Np, 1);
for i = 1 : size(Centers,1)
    for j = 1 : Np
        x = normrnd(Centers(i,1),0.25);
        y = normrnd(Centers(i,2),0.25);
        X(count,:) = [x y];
        C(count) = i;
        count = count + 1;
    end
end

scatter(X(:,1),X(:,2),'.');
[M, N] = size(X);

%% Setting for DA parameters

K = 8; Tmin = 0.0005; alpha = 0.99; PERTURB = 0.0001; STOP = 1e-2;
T = 80; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
rho = Px;

T_P = zeros(K,K,M);
for j = 1 : K
    for k = 1 : K
        if j ~= k
            T_P(k,j,:) = 2/(K*K);
        end
    end
    T_P(j,j,:) = (K-2)/K;
end


% Preallocate 3D matrix: p_l_given_ji(l, j, i)
p_l_given_ji = zeros(K, K, M);

for j = 1:K
    for i = 1:M
        % Initialize unnormalized probabilities
        probs = zeros(1, K);
        for l = 1:K
            if l == j
                probs(l) = rand * 0.6 + 0.6;  % Strong preference for l == j
            else
                probs(l) = rand * 0.05;       % Small values for l ≠ j
            end
        end

        % Normalize to sum to 1
        probs = probs / sum(probs);

        % Store in matrix
        p_l_given_ji(:, j, i) = probs(:);
    end
end

T_P = p_l_given_ji;


v = VideoWriter('my_simulation_video_With_TP.mp4', 'MPEG-4');  % or 'Motion JPEG AVI'
v.FrameRate = 10;  % frames per second
open(v);
while T >= Tmin
    L_old = inf;
    disp(T);
    while 1
        [D,D_Act] = distortion(X,Y,M,N,K,T_P);
        num = exp(-D/T);
        den = repmat(sum(num,2),[1 K]);
        P = num./den;
        Py = P'*Px;

        for l = 1:K
            % numerator = zeros(1, N);
            % denominator = 0;
            % 
            % for i = 1:M
            %     for j = 1:K
            %         w = (1/M) * P(i,j) * T_P(l, j, i);
            %         numerator = numerator + w * X(i, :);
            %         denominator = denominator + w;
            %     end
            % end

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
    T = T*alpha;
    scatter(X(:,1),X(:,2),'.'); hold on;
    scatter(Y(:,1),Y(:,2),'d','filled'); title(T); hold off;
    frame = getframe(gcf);       % gcf = current figure
    writeVideo(v, frame);  
end
close(v);
