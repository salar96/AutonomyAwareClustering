function s = ShortestPath_reset(action_count, state_count)
    s = randi(state_count);
    while s == action_count
        s = randi(action_count);
    end
end