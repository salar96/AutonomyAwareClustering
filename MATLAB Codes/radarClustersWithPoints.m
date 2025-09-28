function radarClustersWithPoints(X, cluster_labels, k, dimLabels, useMedoid, maxPerCluster)
%RADARCLUSTERSWITHPOINTS 7D radar plot with faint per-point lines (coloured by cluster)
%   X: N×7 data
%   cluster_labels: N×1 integers (cluster IDs, not necessarily 1..k)
%   k: number of clusters
%   dimLabels: 1×7 cell array of axis labels
%   useMedoid: true → medoid; false → centroid
%   maxPerCluster: max number of per-cluster points to plot (default 200)

    if nargin < 3 || isempty(k), k = numel(unique(cluster_labels)); end
    if nargin < 4 || isempty(dimLabels), dimLabels = arrayfun(@(i) sprintf('d%d',i),1:7,'UniformOutput',false); end
    if nargin < 5, useMedoid = false; end
    if nargin < 6, maxPerCluster = 200; end
    assert(size(X,2)==7, 'Expecting 7D data');

    % Min–max scale to [0,1]
    %mins = min(X,[],1); maxs = max(X,[],1);
    %Xs = (X - mins) ./ max(maxs - mins, 1e-12);
    Xs = X;
    % Remap labels to 1..k
    u = unique(cluster_labels(:)');
    lab = zeros(size(cluster_labels));
    for t=1:numel(u), lab(cluster_labels==u(t)) = t; end

    % Representatives
    reps = nan(k,7);
    for c = 1:k
        Xc = Xs(lab==c, :);
        if isempty(Xc), continue; end
        if useMedoid
            D = pdist2(Xc, Xc, 'euclidean');
            [~,idx] = min(sum(D,2));
            reps(c,:) = Xc(idx,:);
        else
            reps(c,:) = mean(Xc,1);
        end
    end

    % Radar geometry
    D = 7;
    theta = linspace(0, 2*pi, D+1); theta(end)=theta(1);
    figure('Color','w'); ax = polaraxes; hold(ax,'on');
    rlim(ax,[0 1]); rticks(ax,[0.2 0.4 0.6 0.8]);
    thetaticks(rad2deg(theta(1:end-1))); thetaticklabels(dimLabels);

    % Define distinct colours for clusters
    cmap = lines(k);

    % Faint per-point polylines (coloured by cluster)
    for c=1:k
        Xc = Xs(lab==c,:);
        if isempty(Xc), continue; end
        if size(Xc,1) > maxPerCluster
            idx = randperm(size(Xc,1), maxPerCluster);
            Xc = Xc(idx,:);
        end
        for r=1:size(Xc,1)
            vals = [Xc(r,:) Xc(r,1)];
            polarplot(ax, theta, vals, 'LineWidth', 0.5, ...
                'Color', [cmap(c,:) 0.5]); % faint cluster colour
        end
    end

    % Bold representative per cluster
    for c=1:k
        if any(isnan(reps(c,:))), continue; end
        vals = [reps(c,:) reps(c,1)];
        polarplot(ax, theta, vals, 'LineWidth', 2, 'Color', cmap(c,:));
    end

    title('7D Radar: clusters (faint coloured lines + bold representatives)');
    legend(arrayfun(@(c) sprintf('Cluster %d',c), 1:k, 'UniformOutput', false), ...
           'Location','northeastoutside');
end
