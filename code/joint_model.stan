// Step 7 of the household PC model: one Bayesian probit on every CE unit, 1994-2013, with the threshold structure
// of the second stage inside it (driver: joint_model.py).
//
//   Pr(own_i = 1) = Phi(beta_t (x_i - (tau_t - 10)) + gamma_g(i),t + z_i' delta_t),   t = t(i)
//   tau_t ~ Normal(a + b ell_t + d (t - 1994), omega^2)
//
// x_i = corrected log long-run spending minus 10; z_i = traits centred at their pooled means; gamma_g,t = geography
// effects with sum_g pi_g gamma_g,t = 0 (pi = pooled group shares; the effect of group `ref` is set by the constraint),
// so tau_t is the threshold at the pooled geography and traits, as in the first stage.
// The likelihood is weighted by the survey weights (a pseudo-likelihood). Phi is computed with Phi_approx's logistic
// approximation (largest error in probability 0.00014). The years are summed in parallel with reduce_sum.
//
// Sampling coordinates (a change of variables; the posterior is the one above). In year t the linear predictor is
// X_t coef_t, with X_t = [1, x, geography columns, traits] and coef_t = (c_t, beta_t, free gamma_t, delta_t), where
// c_t = beta_t (10 - tau_t). The sampler moves theta_t, with coef_t = coef0_t + R_t^-1 theta_t: R_t is the triangular
// factor of the thin QR decomposition of diag(sqrt(w)) X_t stacked on the square root of the prior precision of the
// geography and trait effects, so R_t'R_t = X_t' W X_t + that precision (full rank even when a group has no units in
// a year) and theta_t is close to uncorrelated; coef0_t is year t's penalized fit on its own (the first stage with
// this link), so theta_t = 0 is the two-step solution. Likewise
// (a, b, d) = abd0 + R_H^-1 theta_H, with R_H from the thin QR of [1, ell_t, t - 1994] (Stan User's Guide, "QR
// reparameterization"). Both maps are linear, so their Jacobians are constant; the map from c_t to tau_t adds
// -log(beta_t) to the log density.
functions {
  real years_ll(array[] int slice, int first, int last, matrix Q, vector eta0, vector sgn, vector w,
                array[] int start, array[] int len, array[] vector theta) {
    real lp = 0;
    for (j in 1:size(slice)) {
      int t = slice[j];
      int s = start[t];
      int n = len[t];
      vector[n] eta = segment(eta0, s, n) + block(Q, s, 1, n, cols(Q)) * theta[t];
      // log Pr(y) = log inv_logit((2y - 1) u), u = Phi_approx's cubic in eta
      lp += dot_product(segment(w, s, n),
                        log_inv_logit(segment(sgn, s, n) .* (0.07056 * eta .* eta .* eta + 1.5976 * eta)));
    }
    return lp;
  }
}
data {
  int<lower=1> N;
  int<lower=1> T;
  int<lower=2> G;
  int<lower=0> K;
  array[T] int<lower=1> start;
  array[T] int<lower=1> len;
  array[N] int<lower=1, upper=G> geo;
  vector[N] x;
  matrix[N, K] Z;
  vector<lower=0, upper=1>[N] y;
  vector<lower=0>[N] w;
  vector[T] ell;
  vector[T] trend;
  vector<lower=0>[G] pi;
  int<lower=1, upper=G> ref;
  array[T] vector[1 + G + K] coef0;
  vector[3] abd0;
  int<lower=1> grainsize;
}
transformed data {
  int P = 1 + G + K;    // intercept, slope, G - 1 free geography effects, K traits
  array[T] int years = linspaced_int_array(T, 1, T);
  array[G - 1] int free;
  array[G] int pos = rep_array(0, G);    // position of each free group among the free effects
  {
    int j = 1;
    for (g in 1:G) {
      if (g != ref) {
        free[j] = g;
        pos[g] = j;
        j += 1;
      }
    }
  }
  vector[G - 1] pi_free = pi[free];
  vector[N] sgn = 2 * y - 1;
  matrix[G + K, P] Lp = rep_matrix(0, G + K, P);    // square root of the prior precision of the effects
  for (j in 1:(G - 1)) {
    Lp[j, 2 + j] = 1;
  }
  Lp[G, 3:(G + 1)] = -(pi_free / pi[ref])';
  for (k in 1:K) {
    Lp[G + k, G + 1 + k] = inv(2.5);
  }
  matrix[N, P] Q;
  vector[N] eta0;
  array[T] matrix[P, P] Rinv;
  for (t in 1:T) {
    int s = start[t];
    int n = len[t];
    matrix[n, P] X = rep_matrix(0, n, P);
    X[ : , 1] = rep_vector(1, n);
    X[ : , 2] = segment(x, s, n);
    for (i in 1:n) {
      int g = geo[s + i - 1];
      if (g == ref) {
        X[i, 3:(G + 1)] = -(pi_free / pi[ref])';
      } else {
        X[i, 2 + pos[g]] = 1;
      }
    }
    if (K > 0) {
      X[ : , (G + 2):P] = block(Z, s, 1, n, K);
    }
    Rinv[t] = inverse(qr_thin_R(append_row(diag_pre_multiply(sqrt(segment(w, s, n)), X), Lp)));
    Q[s:(s + n - 1), : ] = X * Rinv[t];
    eta0[s:(s + n - 1)] = X * coef0[t];
  }
  matrix[T, 3] H = append_col(append_col(rep_vector(1, T), ell), trend);
  matrix[3, 3] RHinv = inverse(qr_thin_R(H) / sqrt(T - 1));
}
parameters {
  vector[3] theta_H;
  real<lower=0> omega;
  array[T] vector[P] theta;
}
transformed parameters {
  real a;
  real b;
  real d;
  vector[T] tau;
  vector<lower=0>[T] beta;
  {
    vector[3] abd = abd0 + RHinv * theta_H;
    a = abd[1];
    b = abd[2];
    d = abd[3];
  }
  for (t in 1:T) {
    real c = coef0[t][1] + Rinv[t][1] * theta[t];
    beta[t] = coef0[t][2] + Rinv[t][2] * theta[t];
    tau[t] = 10 - c / beta[t];
  }
}
model {
  a ~ normal(10, 2);
  b ~ normal(0.5, 1);
  d ~ normal(0, 0.2);
  omega ~ normal(0, 0.2);
  tau ~ normal(a + b * ell + d * trend, omega);
  beta ~ normal(0.8, 0.5);
  target += -sum(log(beta));    // Jacobian of c_t -> tau_t
  for (t in 1:T) {
    vector[P - 2] rest = coef0[t][3:P] + Rinv[t][3:P, : ] * theta[t];    // free geography and trait effects
    vector[G] gamma;
    gamma[free] = rest[1:(G - 1)];
    gamma[ref] = -dot_product(pi_free, rest[1:(G - 1)]) / pi[ref];
    target += normal_lpdf(gamma | 0, 1);
    if (K > 0) {
      target += normal_lpdf(rest[G:(P - 2)] | 0, 2.5);
    }
  }
  target += reduce_sum(years_ll, years, grainsize, Q, eta0, sgn, w, start, len, theta);
}
