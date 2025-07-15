function P = association_weights(D, T, K)
    num = exp(-D/T);
    den = repmat(sum(num,2),[1 K]);
    P = num./den;
end