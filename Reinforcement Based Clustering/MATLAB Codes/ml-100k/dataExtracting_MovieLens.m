% Step 1: Read u.data file (ratings)
ratings = readmatrix('u.data', 'FileType', 'text');
user_ids = ratings(:,1);
item_ids = ratings(:,2);
rating_vals = ratings(:,3);
timestamps = ratings(:,4);

% Step 2: Read u.item file
fid = fopen('u.item');
C = textscan(fid, '%d %s %s %s %s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d', ...
             'Delimiter', '|', 'ReturnOnError', false);
fclose(fid);

movie_ids = C{1};
release_dates_str = C{3};
num_movies = length(movie_ids);

% Parse release years
release_years = zeros(num_movies, 1);
for i = 1:num_movies
    try
        d = datetime(release_dates_str{i}, 'InputFormat', 'dd-MMM-yyyy', 'Locale', 'en_US');
        release_years(i) = year(d);
    catch
        release_years(i) = NaN;
    end
end

% Extract genre flags: 19 binary columns
genre_flags = cell2mat(C(6:end));  % Now correctly interpreted as numeric

% Step 3: Build user feature matrix
user_list = unique(user_ids);
num_users = length(user_list);
features = zeros(num_users, 7);  % 7 features

% Feature index constants
WATCHTIME = 1;
GENRESWATCHED = 2;
TRENDINGPCT = 3;
BINGEFREQ = 4;
COMPLETIONRATE = 5;
TIER = 6;
LATENIGHTPCT = 7;

avg_movie_length = 90;
trending_threshold = 1995;

for i = 1:num_users
    uid = user_list(i);
    idx = user_ids == uid;

    user_movies = item_ids(idx);
    user_ratings = rating_vals(idx);
    user_times = timestamps(idx);

    % Feature 1: WatchTime
    features(i, WATCHTIME) = length(user_movies) * avg_movie_length;

    % Feature 2: GenresWatched
    genre_mat = genre_flags(user_movies, :);
    features(i, GENRESWATCHED) = sum(any(genre_mat, 1));

    % Feature 3: TrendingContentPct
    release_subset = release_years(user_movies);
    features(i, TRENDINGPCT) = sum(release_subset >= trending_threshold) / length(user_movies);

    % Feature 4: BingeFreq (3+ ratings in a day)
    days = floor(user_times / 86400);  % Convert to days
    binge_counts = histcounts(days, 'BinMethod', 'integers');
    features(i, BINGEFREQ) = sum(binge_counts >= 3);

    % Feature 5: CompletionRate (ratings >= 4)
    features(i, COMPLETIONRATE) = sum(user_ratings >= 4) / length(user_ratings);

    % Feature 6: Subscription Tier (based on watch time)
    wt = features(i, WATCHTIME);
    if wt > 10000
        tier = 3;  % Premium
    elseif wt > 5000
        tier = 2;  % Standard
    else
        tier = 1;  % Basic
    end
    features(i, TIER) = tier;

    % Feature 7: LateNightWatchPct
    hours = mod(user_times / 3600, 24);
    features(i, LATENIGHTPCT) = sum(hours >= 23 | hours < 4) / length(hours);
end

% Step 4: Create table and save
feature_table = array2table(features, ...
    'VariableNames', {'WatchTime', 'GenresWatched', 'TrendingContentPct', ...
                      'BingeFreq', 'CompletionRate', 'SubscriptionTier', ...
                      'LateNightWatchPct'});
feature_table.UserID = user_list;
feature_table = movevars(feature_table, 'UserID', 'Before', 1);

writetable(feature_table, 'streaming_users_from_movielens.csv');
