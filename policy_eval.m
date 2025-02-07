function Q = policy_eval(gamma,P,AvC, state_count, action_count)
V = zeros(state_count, 1);
V_old = V;
count1 = 1;
while count1 <= 1000    
    Q = (AvC + gamma*squeeze(sum(P.*repmat(V,[1,state_count, action_count]),1)));
    V = min(Q,[],2);
    if (norm(V-V_old) < 0.001)
        break;
    else
        V_old = V;
    end
    count1 = count1 + 1;
end
end