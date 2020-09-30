function [grad_output, dA, cost_mini_batch] = LSTM_cost_alignment(Y, Y_pred, A, ...
    param, lambda)

[~, m_trials, t_time] = size(A);

dW_y = zeros(size(param.W_y));
db_y = zeros(size(param.b_y));

dA = zeros(size(A));
cost_mini_batch = 1/2/t_time/m_trials*sum(sum(sum((Y_pred-Y).^2,1), ...
    2),3);

for t = 1:t_time
    dY_pred = Y_pred(:,:,t)-Y(:,:,t); 
    dA(:,:,t) = transpose(param.W_y)*dY_pred;
    dW_y = dW_y + 1/m_trials*(dY_pred*transpose(A(:,:,t)));
    db_y = db_y + 1/m_trials*sum(dY_pred,2);
end

grad_output.dW_y = dW_y + lambda/m_trials*param.W_y;
grad_output.db_y = db_y;

end