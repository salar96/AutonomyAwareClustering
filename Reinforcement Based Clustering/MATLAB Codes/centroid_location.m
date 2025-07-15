function Y = centroid_location(K, T_P, M, P, X)
    for l = 1:K
        T_slice = squeeze(T_P(l, :, :))';   
        W = (1/M) * P .* T_slice;           
        row_weights = sum(W, 2);            
        numerator = row_weights' * X;
        denominator = sum(row_weights);
        Y(l, :) = numerator / denominator;
    end
end