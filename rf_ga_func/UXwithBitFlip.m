function Offspring = UXwithBitFlip(parent1, parent2, crossover_rate)

    %% default parameter setting
    if nargin < 3
        crossover_rate = 0.9;
    end
    
    [num, len] = size(parent1);
    crossover_rand = rand(num, 1);

    choose_id = logical(round(rand(num, len)));
    Offspring = parent2;
    Offspring(choose_id) = parent1(choose_id);
    Offspring(crossover_rand > crossover_rate, :) = parent1(crossover_rand > crossover_rate, :);

    Offspring = BitFlip(Offspring);           

end