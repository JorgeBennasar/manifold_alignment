function [A, Y_pred, cache] = LSTM_forward_prop_alignment(X, param)

[~, m_trials, t_time] = size(X);
[n_output, n_hidden] = size(param.W_y);

A = zeros(n_hidden, m_trials, t_time);
C = zeros(n_hidden, m_trials, t_time);

A_next = zeros(n_hidden, m_trials);
C_next = zeros(n_hidden, m_trials);
Y_pred = zeros(n_output, m_trials, t_time);
for t = 1:t_time
    A_prev = A_next;
    C_prev = C_next; 
    X_t = X(:,:,t); 
    concat = [A_prev; X_t];
    Ft = 1./(1+exp(-(param.W_f)*concat + param.b_f)); 
    It = 1./(1+exp(-(param.W_i)*concat + param.b_i));
    Cct = tanh((param.W_c)*concat + param.b_c);
    C_next = C_prev.*Ft + It.*Cct; 
    C(:,:,t) = C_next;
    Ot = 1./(1+exp(-(param.W_o)*concat + param.b_o));
    A_next = Ot.*tanh(C_next);
    A(:,:,t) = A_next;
    Y_pred(:,:,t) = param.W_y*A_next + param.b_y;
    cache(t).A_next = A_next;
    cache(t).A_prev = A_prev;
    cache(t).C_next = C_next;
    cache(t).C_prev = C_prev;
    cache(t).Ft = Ft;
    cache(t).It = It;
    cache(t).Cct = Cct;
    cache(t).Ot = Ot;
end

end