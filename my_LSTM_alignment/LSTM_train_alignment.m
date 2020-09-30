function [param, cost_train] = LSTM_train_alignment(X, Y, mini_batch_size, ...
    num_epochs, n_hidden, beta_1, beta_2, epsilon, learning_rate, ...
    optimization, lambda, stop_condition, connectivity, network_model, links, ...
    correlation_reg)

rng('default');
rng shuffle;

[n_input, m_trials, t_time] = size(X);
[n_output, ~, ~] = size(Y);

[param, v, s] = LSTM_initialization_alignment(n_input, n_hidden, n_output, ...
        connectivity, network_model, links);

num_batch = fix(m_trials/mini_batch_size);
t = 0;
costs = [];

for i = 1:num_epochs
    idx = randperm(m_trials);
    costs_epoch = [];

    for j = 1:num_batch
        mini_batch_X = zeros(n_input, mini_batch_size, t_time);
        mini_batch_Y = zeros(n_output, mini_batch_size, t_time);
        for k = 1:mini_batch_size
            mini_batch_X(:,k,:) = X(:,idx((j-1)*mini_batch_size+k),:);
            mini_batch_Y(:,k,:) = Y(:,idx((j-1)*mini_batch_size+k),:);
        end

        [A, Y_pred, cache] = LSTM_forward_prop_alignment(mini_batch_X, param);
        [grad_output, dA, cost_mini_batch] = LSTM_cost_alignment(mini_batch_Y, ...
            Y_pred, A, param, lambda);
        grad_hidden = LSTM_backward_prop_alignment(mini_batch_X, dA, cache, ...
            param, lambda, A, correlation_reg);
        t = t + 1;
        [param, v, s] = LSTM_update_param_alignment(param, grad_hidden, ...
            grad_output, v, s, beta_1, beta_2, t, epsilon, ...
            learning_rate, optimization);

        costs_epoch = [costs_epoch cost_mini_batch];
    end

    costs = [costs mean(costs_epoch)];
    aux = linspace(1,length(costs),length(costs));
    figure(30);
    plot(aux,costs,'Color',[0 0.5 1],'LineWidth',2);
    axis tight;
    title('Training cost');
    ylabel('cost');
    xlabel('epoch');
    
    if mean(costs_epoch) <= stop_condition
        break;
    end
end

cost_train = costs(end);

end