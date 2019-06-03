function  y = calc_default_tsne_2dim(seed, data)

    rng(seed);

    if size(data, 1) > 200
        if size(data, 2) > 50
            y = tsne(data, 'NumPCAComponents', 50, 'NumDimensions', 2, 'Standardize', true, 'Perplexity', 50);
        else
            y = tsne(data, 'NumDimensions', 2, 'Standardize', true, 'Perplexity', 50);
        end
    else
        if size(data, 2) > 50
            y = tsne(data, 'NumPCAComponents', 50, 'NumDimensions', 2, 'Standardize', true);
        else
            y = tsne(data, 'NumDimensions', 2, 'Standardize', true);
        end
    end
end