% %% Coding the environment for the FLPO learning problem
% 
% function [next_state, cost, done] = ShortestPath_stoch2(state, action, para, resource_count, action_count, state_count)
%     if state == action_count
%         cost = 0;
%         next_state = state;
%         done = 1;
%         return;
%     end
%     if state == action
%         cost = 100;
%         next_state = state;
%     elseif ismember(state, action_count+1:state_count) && action == action_count
%         cost = 100;
%         next_state = state;
%     else
%         prob = rand;
%         if prob < 0.1
%             % take random action
%             action = randi(resource_count);
%             cost = norm(para(state,:) - para(action,:))^2;
%             next_state = action;
%         else
%             % take proper action
%             cost = norm(para(state,:) - para(action,:))^2;
%             next_state = action;
%         end
%     end
%     if next_state == action_count
%         done = 1;
%     else 
%         done = 0;
%     end
% end

%% Coding the environment for the FLPO learning problem

function [next_state, cost, done] = ShortestPath_stoch2(state, action, para, resource_count, action_count, state_count)
    my_inf = 10000;
    % if state == action_count % at dest, the action does not matter, the cost is zero and we remain there
    %     cost = 0;
    %     next_state = state;
    %     done = 1;
    %     return;
    % end
    % %if state == action
    % %    cost = 100;
    % %    next_state = state;
    % if ismember(state, action_count+1:state_count) && action == action_count
    %     cost = my_inf; % n2d
    %     next_state = state; %remain there
    % else
    %     prob = rand;
    %     if prob < 0  % change back to <= 0.1
    %         % take random action
    %         %action = randi(resource_count);
    %         action = 1;
    %         if state == action
    %             cost = 100;
    %         else
    %             cost = norm(para(state,:) - para(action,:))^2; 
    %         end
    %         next_state = action;
    %     else
    %         % take proper action
    %         if state == action
    %             cost = 100;
    %         else
    %             cost = norm(para(state,:) - para(action,:))^2;
    %         end
    %         next_state = action;
    %     end
    % end
    % if next_state == action_count
    %     done = 1;
    % else 
    %     done = 0;
    % end
    
    if state == action_count %at dest
        next_state = state;
        cost = 0;
        done=1;
        
    elseif ismember(state,1:resource_count) %at f
        if ismember(action,action_count+1:state_count) %f2n
            cost = my_inf;
            next_state = state;
            done = 0;
        elseif action==action_count %f2d
            cost = 0;
            next_state = action;
            done=1;
        else %f2f
            cost=0;
            next_state = action;
            done=0;
        end
        
    else % at n
        if ismember(action,action_count+1:state_count) %n2n
            cost=my_inf;
            next_state=state;
            done=0;
        elseif action==action_count %n2d
            cost=my_inf;
            next_state=state;
            done=0;
        else %n2f
            cost = norm(para(state,:) - para(action,:))^2; 
            next_state = action;
            done = 1;
        end
    end

end