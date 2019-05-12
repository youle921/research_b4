function y = get_colortable(x)

y = zeros(size(x, 1), 3);

tmp = x(:, 1) - min(x(:, 1));
tmp = (tmp / max(tmp)) * 2 - 1;
g = 20;
offset_rb = 0.25;
offset_g = 0.7;

%red
y(:, 1) = sigmoid(tmp, g, -offset_rb);
%green
y(:, 2) = sigmoid(tmp, g, offset_g) + (1 - sigmoid(tmp, g, -offset_g)) - 1;

%blue
y(:, 3) = 1 - sigmoid(tmp, g, offset_rb);

end

function y = sigmoid(x, gain, offset) 
    y = (tanh((gain * (x + offset)) / 2) + 1) / 2;
end

%offset -0.2 0.6