function flag = nn_check(net,Xs,Y,j_idx)

    M = size(Xs,1);  Yvec = Y(:); 
    K = size(Y,1);
    for i = 1 : M
       j = j_idx(i);
       onehot = zeros(K,1); onehot(j) = 1;
       u = [Xs(i,:).'; onehot; Yvec];
       disp([net(u) norm(Xs(i,:)-Y(j,:))^2])
    end
    flag = 0;
end