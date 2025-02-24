library("cmdstanr")
library("bayesplot")
library("posterior")
schools_dat <- list(J=8, y = c(28,8,-4,7,-1,1,18,12),
                    sigma = c(15,10,16,11,9,11,10,18))

 
mod <- cmdstan_model("schools.stan")
fit <- mod$sample(
  data = schools_dat,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  refresh = 500 # print update every 500 iters
)
draws_arr <- fit$draws() # or format="array"
str(draws_arr)
mcmc_hist(fit$draws("mu"))
