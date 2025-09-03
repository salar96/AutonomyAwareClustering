function check_grad_Yl_fitnet(net, X, J, Y, l, i, cols, eps0)
    if nargin<8, eps0 = 1e-1; end
    if nargin<7, cols = randperm(size(Y,2), min(5,size(Y,2))); end
    if nargin<6, i = randi(size(X,1)); end
    K = size(Y,1);

    dYl = grad_fitnet2(net, X, J, Y, Y(l,:), l);
    gA = dYl(i,:);

    for c = cols
        Yp = Y; Ym = Y;
        Yp(l,c) = Yp(l,c) + eps0;
        Ym(l,c) = Ym(l,c) - eps0;

        up = [X(i,:), onehot(J(i),K), Yp(:)'];
        um = [X(i,:), onehot(J(i),K), Ym(:)'];

        yp = net(up');   % IMPORTANT: let net handle its own processing
        ym = net(um');
        gN = (yp - ym)/(2*eps0);
        fprintf('c=%d  analytic=% .6e  numeric=% .6e  diff=% .2e\n', c, gA(c), gN, abs(gA(c)-gN));
    end
end
function oh = onehot(j,K), oh=zeros(1,K); oh(j)=1; end
