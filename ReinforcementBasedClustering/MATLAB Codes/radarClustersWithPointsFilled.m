function radarClustersWithPointsFilled(X, cluster_labels, k, dimLabels, useMedoid, maxPerCluster, fillAlpha, pointAlpha)
%RADARCLUSTERSWITHPOINTSFILLED
% 7D radar plot with (i) faint per-point polylines coloured by cluster,
% and (ii) bold representative polygons (centroid/medoid) with translucent fill.
%
% X              : N×7 data
% cluster_labels : N×1 integers (cluster IDs; need not be 1..k)
% k              : number of clusters (optional; inferred if omitted)
% dimLabels      : 1×7 cell array of axis labels (optional)
% useMedoid      : true → medoid; false → centroid (optional; default false)
% maxPerCluster  : cap per-cluster lines to avoid clutter (optional; default 200)
% fillAlpha      : alpha for filled representative polygon (optional; default 0.18)
% pointAlpha     : alpha for faint per-point lines (optional; default 0.15)

    if nargin < 3 || isempty(k), k = numel(unique(cluster_labels)); end
    if nargin < 4 || isempty(dimLabels), dimLabels = arrayfun(@(i) sprintf('d%d',i),1:7,'UniformOutput',false); end
    if nargin < 5, useMedoid = false; end
    if nargin < 6, maxPerCluster = 200; end
    if nargin < 7, fillAlpha = 0.18; end
    if nargin < 8, pointAlpha = 0.15; end
    assert(size(X,2)==7, 'Expecting 7D data');

    % ----- Min–max scale to [0,1] -----
    mins = min(X,[],1); maxs = max(X,[],1);
    Xs = (X - mins) ./ max(maxs - mins, 1e-12);

    % ----- Remap labels to 1..k -----
    u = unique(cluster_labels(:)');
    lab = zeros(size(cluster_labels));
    for t=1:numel(u), lab(cluster_labels==u(t)) = t; end

    % ----- Representatives (centroid or medoid) -----
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

    % ----- Radar geometry -----
    D = 7;
    theta = linspace(0, 2*pi, D+1); theta(end)=theta(1);

    % ----- Create polar grid axes -----
    f = figure('Color','w'); %#ok<NASGU>
    ax = polaraxes; hold(ax,'on');
    rlim(ax,[0 1]); rticks(ax,[0.2 0.4 0.6 0.8]);
    thetaticks(rad2deg(theta(1:end-1))); thetaticklabels(dimLabels);

    % ----- Overlay a transparent Cartesian axes for filled patches -----
    axCart = axes('Position', ax.Position, 'Color','none'); hold(axCart,'on');
    axis(axCart,'equal'); axis(axCart,[-1 1 -1 1]); axis(axCart,'off');

    % ----- Colours -----
    cmap = lines(k);

    % ----- Faint per-point polylines, coloured by cluster -----
    for c=1:k
        Xc = Xs(lab==c,:);
        if isempty(Xc), continue; end
        if size(Xc,1) > maxPerCluster
            idx = randperm(size(Xc,1), maxPerCluster);
            Xc = Xc(idx,:);
        end
        for rIdx=1:size(Xc,1)
            rvals = [Xc(rIdx,:) Xc(rIdx,1)];         % close polygon
            polarplot(ax, theta, rvals, 'LineWidth', 0.6, ...
                'Color', [cmap(c,:) pointAlpha]);
        end
    end

    % ----- Bold representatives + translucent fill (via Cartesian overlay) -----
    legHandles = gobjects(1,k);
    for c=1:k
        if any(isnan(reps(c,:))), continue; end
        rvals = [reps(c,:) reps(c,1)];
        % Bold outline on polar axes
        h = polarplot(ax, theta, rvals, 'LineWidth', 2, 'Color', cmap(c,:));
        legHandles(c) = h;

        % Filled patch on Cartesian overlay
        [xFill, yFill] = pol2cart(theta, rvals);
        fill(axCart, xFill, yFill, cmap(c,:), ...
            'FaceAlpha', fillAlpha, 'EdgeColor', 'none');
    end

    title(ax, '7D Radar: coloured per-point lines + filled representative polygons');
    legend(ax, legHandles, arrayfun(@(c) sprintf('Cluster %d',c), 1:k, 'UniformOutput', false), ...
           'Location','northeastoutside');
end
