%% This code implements the DA algorithm.

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
X_d1 = X;
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
X_d2 = X;
%% Constructing Data Set 3
C11 = [-8 -4]; C21 = [4 -4]; C31 = [4 4]; C41 = [-8 4];

rng('default');
Centers = [C11; C21; C31; C41];

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
X_d3 = X;

%% Constructing Data Set 4
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
X_d4 = X;

%% Constructing Data Set 5

C1 = [2,4]; C2 = [4,7]; C3 = [5,5]; C4 = [5,3]; C5 = [4,1];
rng('default');
Centers = [C1; C2; C3; C4; C5];
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
X_d5 = X;

X = X_d1;
scatter(X(:,1),X(:,2),'.');
[M, N] = size(X);

%% Setting for DA parameters

K = 5; Tmin = 0.0005; alpha = 0.99; PERTURB = 0.0001; STOP = 1e-2;
T = 80; Px = (1/M)*ones(M,1); Y = repmat(Px'*X, [K,1]);
rho = Px; beta_cr = 0;
v = VideoWriter('DA_simulation_video_With_TP.mp4', 'MPEG-4');  % or 'Motion JPEG AVI'
v.FrameRate = 10;  % frames per second
open(v);
while T >= Tmin
    L_old = inf;
    while 1
        [D,D_Act] = distortion_DA(X,Y,M,N,K);
        num = exp(-D/T);
        den = repmat(sum(num,2),[1 K]);
        P = num./den;
        Py = P'*Px;
        Y = P'*(X.*repmat(Px,[1 N]))./repmat(Py,[1 N]) + PERTURB*rand(size(Y));
        L = -T*Px'*log(sum(exp(-D/T),2));
        if(norm(L-L_old) < STOP)
            break;
        end
        L_old = L;
    end
    T = T*alpha;
    T_cr = critical_betaDA(X, Y, M, P, rho);
    fprintf('%d %d \n',T,T_cr);
    T = T*alpha;
    scatter(X(:,1),X(:,2),'.'); hold on;
    scatter(Y(:,1),Y(:,2),'d','filled'); title(T); hold off;
    frame = getframe(gcf);       % gcf = current figure
    writeVideo(v, frame);  
end
close(v);
