function [Y_pred, A] = LSTM_predict_alignment(X, param)

[A, Y_aux, ~] = LSTM_forward_prop_alignment(X, param);
Y_pred = Y_aux;

end