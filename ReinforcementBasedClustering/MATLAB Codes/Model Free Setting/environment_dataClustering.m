function [k, dist] = environment_dataClustering(i,j,T_P,K,X,Y)
   
    values = linspace(1,K,K);
    p = T_P(:,j,i);
    k = randsample(values, 1, true, p);
    dist = norm(X(i,:)-Y(k,:))^2;

end