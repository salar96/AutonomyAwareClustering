function generate_clusters(k, N, p, filename)
    % generate_clusters: Generates k clusters of 2D points and saves to a .mat file.
    %
    % Inputs:
    %   k: Number of clusters
    %   N: Approximate number of points in each cluster
    %   p: Percentage variation in the number of points (±p%)
    %   filename: Name of the .mat file to save the data
    %
    % Output:
    %   Saves the generated clusters into a .mat file.

    % Validate inputs
    if k <= 0 || N <= 0 || p < 0 || p > 100
        error('Invalid input parameters. Ensure k > 0, N > 0, and 0 <= p <= 100.');
    end

    % Initialize variables
    data = [];          % To store all points
    labels = [];        % To store cluster labels
    cluster_centers = 10 * rand(k, 2); % Random cluster centers within [0, 10]

    % Generate clusters
    for i = 1:k
        % Determine the number of points in this cluster with ±p% variation
        n_points = round(N * (1 + (rand() - 0.5) * p / 50));

        % Generate points around the cluster center with some random spread
        spread = 1; % Spread of points around the center (adjusted for [0, 10] range)
        points = cluster_centers(i, :) + spread * (rand(n_points, 2) - 0.5);

        % Ensure points are within [0, 10]
        points = max(min(points, 10), 0);

        % Append points and labels
        data = [data; points];
        labels = [labels; i * ones(n_points, 1)];
    end

    % Save the data to a .mat file
    save(filename, 'data', 'labels', 'cluster_centers');
    fprintf('Data saved to %s\n', filename);
end