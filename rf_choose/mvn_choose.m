function choose_id = mvn_choose(data, ratio)

c_num = round(ratio * size(data, 1));

m = mean(data);
sigma = cov(data);

ref_point = mvnrnd(m, sigma, c_num);

choose_id = knnsearch(data, ref_point);
    
end
