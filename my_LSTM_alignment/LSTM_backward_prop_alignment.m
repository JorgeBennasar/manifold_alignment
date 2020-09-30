function grad_hidden = LSTM_backward_prop_alignment(X, dA, cache, param, ...
    lambda, A, correlation_reg)

[n_hidden, m_trials, ~] = size(dA);
[~, ~, t_time] = size(X);

dW_f = zeros(size(param.W_f));
dW_i = zeros(size(param.W_i));
dW_c = zeros(size(param.W_c));
dW_o = zeros(size(param.W_o));
db_f = zeros(size(param.b_f));
db_i = zeros(size(param.b_i));
db_c = zeros(size(param.b_c));
db_o = zeros(size(param.b_o));

dA_prev = zeros(n_hidden,m_trials);
dC_prev = zeros(n_hidden,m_trials);   
for t = t_time:-1:1
    dA_next = dA_prev + dA(:,:,t);  
    dC_next = dC_prev; 

    A_prev = cache(t).A_prev;
    C_next = cache(t).C_next;
    C_prev = cache(t).C_prev;
    Ft = cache(t).Ft;
    It = cache(t).It;
    Cct = cache(t).Cct;
    Ot = cache(t).Ot;

    dot = dA_next.*tanh(C_next);
    dcct = (dA_next.*Ot.*(1-tanh(C_next).^2)+dC_next).*It;
    dit = (dA_next.*Ot.*(1-tanh(C_next).^2)+dC_next).*Cct;
    dft = (dA_next.*Ot.*(1-tanh(C_next).^2)+dC_next).*C_prev;

    dit = dit.*It.*(1-It);
    dft = dft.*Ft.*(1-Ft);
    dot = dot.*Ot.*(1-Ot);
    dcct = dcct.*(1-Cct.^2);

    X_t = X(:,:,t); 
    concat = [A_prev; X_t];

    dW_f = (dW_f + dft*transpose(concat)).*param.G;
    dW_i = (dW_i + dit*transpose(concat)).*param.G;
    dW_c = (dW_c + dcct*transpose(concat)).*param.G;
    dW_o = (dW_o + dot*transpose(concat)).*param.G;
    db_f = db_f + sum(dft,2);
    db_i = db_i + sum(dit,2);
    db_c = db_c + sum(dcct,2);
    db_o = db_o + sum(dot,2);

    d_concat = transpose(param.W_f)*dft + ...
        transpose(param.W_o)*dot + ...
        transpose(param.W_i)*dit + ...
        transpose(param.W_c)*dcct;
    dA_prev = d_concat(1:n_hidden,:);
    dC_prev = (dA_next.*Ot.*(1-tanh(C_next).^2)+dC_next).*Ft;
end

M = zeros(size(A,1),size(A,2)*size(A,3));
for i = 1:size(A,2)
	M(:,(size(A,3)*(i-1)+1):(size(A,3)*i)) = squeeze(A(:,i,:));
end
R_aux = corrcoef(transpose(M));
R = R_aux;
D = [R zeros(size(R,1),size(param.W_f,2)-size(R,2))];

V = sum(param.G(:,1:n_hidden),2);
P_1 = repmat(V,1,n_hidden);
P_2 = ones(n_hidden,size(X,1));
P = [P_1 P_2];

grad_hidden.dW_f = (dW_f + lambda/m_trials*param.W_f); %.*(P.^4); %+ correlation_reg*D.*param.W_f;
grad_hidden.dW_i = (dW_i + lambda/m_trials*param.W_i); %.*(P.^4); %+ correlation_reg*D.*param.W_i;
grad_hidden.dW_c = (dW_c + lambda/m_trials*param.W_c); %.*(P.^4); %+ correlation_reg*D.*param.W_c;
grad_hidden.dW_o = (dW_o + lambda/m_trials*param.W_o); %.*(P.^4); %+ correlation_reg*D.*param.W_o;
grad_hidden.db_f = db_f;
grad_hidden.db_i = db_i;
grad_hidden.db_c = db_c;
grad_hidden.db_o = db_o;
  
end