data {
    int<lower=0> N; //  number of samples per study
    int<lower=0> K; // number of studies
    array[K] int<lower=0, upper = N> y; // observed data
}

parameters {
    ordered[2] alpha; // ordered logits
}

transformed parameters {
    vector[2] p; // probabilities
    p = inv_logit(alpha); // transform logits to probabilities
    // caclualte log prob for each alpha

    matrix[K, 2] lg;

    for (n in 1:K){
        for (s in 1:2){
            lg[n,s] = binomial_logit_lpmf(y[n] | N, alpha[s]); // log prob for each alpha
        }
   }
}


model {
    // Priors
    alpha ~ normal(0, 10); // Normal prior for the ordered logits
    // Likelihood
    
    for (n in 1:K){
        target += log_sum_exp(lg[n]); // log sum exp for the log probabilities
        }
    
}

generated quantities {
    vector[K] s; // predicted group
    for (n in 1:K){
        // calculate the predicted group
       s[n] = exp( lg[n,1] - log_sum_exp(lg[n]));     }
}
 