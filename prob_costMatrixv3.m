function [P, C] = prob_costMatrixv3(para, state_count, action_count)
P = zeros(state_count, state_count, action_count);
C = zeros(state_count, state_count, action_count);

for a = 1 : action_count
    for s = 1 : state_count
        if s == action_count
            P(:,s,a) = 0;
            P(s,s,a) = 1;
        elseif ismember(s,action_count+1:state_count) && a == action_count
            P(s,s,a) = 1;
        else
            if a ~= 1
                P(1,s,a) = 0.1;   
                P(a,s,a) = 0.9;  
            else
                P(1,s,a) = 1;
            end
        end
    end
end
% for s = 1: state_count
%     for s_p = 1 : state_count
%         if s_p == s && s_p ==  action_count % from base to base
%             C(s_p,s,:) = 0;
%         elseif s_p == s && s_p ~= action_count % each node/facility to itself (except for base)
%             C(s_p,s,:) = 100;
%         elseif ismember(s,action_count+1:state_count) && s_p == action_count % node to base
%             C(s_p,s,:) = 100;
%         elseif ismember(s_p,action_count+1:state_count) % anything to node
%             C(s_p,s,:) = 100;
%         elseif ismember(s,1:action_count-1) && s_p == action_count % facility to base
%             C(s_p,s,:) = 0;
%         elseif ismember(s,1:action_count-1) && ismember(s_p,1:action_count-1) && s_p ~= s
%             C(s_p,s,:) = 0;
%         else
%             C(s_p,s,:) = norm(para(s_p,:) - para(s,:))^2;
%         end
%     end
% end
for i = 1:state_count
    for j = 1:state_count
        C(i, j,:) = norm(para(i,:) - para(j,:))^2;
    end
end

% C(1:state_count+1:end,:) = 100; % self hopping

C(repmat(logical(eye(state_count)), 1, 1, action_count)) = 100;

% Step 2: f2f
C(1:action_count, 1:action_count,:) = 0;

% Step 3: all to nodes
C(action_count+1:end, :,:) = 100;

% Step 4: Modify the action_count-th row
C(action_count, 1:action_count,:) = 0; % f and d to d
C(action_count, action_count+1:end,:) = 100; % node to d

% Step 5: Modify the action_count-th column
C(1:action_count-1, action_count,:) = 100; % d to f
end