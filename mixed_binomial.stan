data {
    int<lower=0> N; //  number of samples per patient
    int<lower=0> K; // number of patients
    array[K] int<lower=0, upper = N> y; // observed data (counts)
}

parameters {
    ordered[2] alpha; // ordered logits
}

transformed parameters {
    vector[2] p; // probabilities from logits 
    p = inv_logit(alpha); // transform logits to probabilities
    // caclualte log prob for each alpha

   // each patient is in one or the other group (well or sick). We assume 
   // equal prior for each possibility. We then calculate teh log prob for each 
   // patient for each possibility

    matrix[K, 2] lp; // log probabilities, one for each possible s for each patient.

    for (n in 1:K){
        for (s in 1:2){
            lp[n,s] = binomial_logit_lpmf(y[n] | N, alpha[s]);  
        }
   }
}


model {
    // Prior on alphas. 
    alpha ~ normal(0, 10); // Normal prior for the ordered logits

    // Likelihood is now just the sum of the probabilities for each patient,
    // marginalized over the two groups.
    // We use the log sum exp trick to avoid numerical underflow.
    
    for (n in 1:K){
        target += log_sum_exp(lp[n]); // log sum exp for the log probabilities
        }
    
}

generated quantities {
    // after the fit we can use Bayes rule directly (normalizing by hand) to 
    // compute the posterior probabilities for each patient being in each group.
    vector[K] s; // predicted group
    for (n in 1:K){
        // calculate the predicted group
       s[n] = exp( lp[n,1] - log_sum_exp(lp[n]));     }
}
 