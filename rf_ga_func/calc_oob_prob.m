function oob_probability = calc_oob_prob(data_num, oob_ratio)

picked_num = round(log(oob_ratio) / log(1 - 1 / data_num));
oob_probability = picked_num / data_num;

end
