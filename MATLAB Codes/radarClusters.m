function radarClusters(X, cluster_labels, k, dimLabels, useMedoid, plotTitle)
% X               : N×7 data
% cluster_labels  : N×1 integers in {0,1,2} or {1,2,3}
% k               : number of clusters (e.g., 3)
% dimLabels       : 1×7 cell array of axis labels
% useMedoid       : logical (true -> medoid, false -> mean)
% plotTitle       : string

    assert(size(X,2) == 7, 'X must be N×7');
    if nargin<6, plotTitle = "Cluster Profiles (Radar)"; end
    if nargin<5, useMedoid = false; end

    % --- Min-Max scale each feature to [0,1] for comparability
    mins = min(X,[],1);
    maxs = max(X,[],1);
    Xs = (X - mins) ./ max( (maxs - mins), 1e-12 );
    %Xs = X;

    % --- Get one 1×7 vector per cluster (mean or medoid)
    cent = zeros(k,7);
    for c = 1:k
        mask = (cluster_labels == c) | (cluster_labels == c-1); % handles 1..k or 0..k-1
        Xc = Xs(mask, :);
        if isempty(Xc)
            warning('Cluster %d is empty. Filling with NaNs.', c);
            cent(c,:) = nan(1,7);
            continue;
        end
        if useMedoid
            % pairwise distances and pick medoid
            D = pdist2(Xc, Xc, 'euclidean');
            [~, idx] = min(sum(D,2));
            cent(c,:) = Xc(idx,:);
        else
            cent(c,:) = mean(Xc,1);
        end
    end

    % --- Radar plot
    D = 7;
    theta = linspace(0, 2*pi, D+1); theta(end) = theta(1);
    figure('Color','w');
    ax = polaraxes; hold(ax,'on');
    rlim(ax, [0 1]);
    rticks(ax,[0.2 0.4 0.6 0.8]);
    thetaticks(rad2deg(theta(1:end-1)));
    if nargin >= 4 && ~isempty(dimLabels)
        thetaticklabels(dimLabels);
    else
        thetaticklabels(arrayfun(@(i) sprintf('d%d',i), 1:D, 'UniformOutput', false));
    end

    legends = strings(1,k);
    for c = 1:k
        vals = cent(c,:);
        valsClosed = [vals, vals(1)];
        polarplot(ax, theta, valsClosed, 'LineWidth', 2);
        legends(c) = sprintf('Cluster %d', c);
    end
    title(plotTitle);
    legend(legends, 'Location','northeastoutside');
end
