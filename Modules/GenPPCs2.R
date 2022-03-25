SimSubjBM2 <- function(sdf, subj_traces, n_samples=100, set_ind_a=1) {
  ### For one subject, samples traces, construct parameters, and generate sims ###
  
  n_samples <- min(nrow(sdf), n_samples)
  idx <- unique(sdf$subj_idx)
  if (length(idx) > 1) stop("Error in extracting subject pars: trying to pull for more than one subject.")
  
  par_vals <- subj_traces[grep(unique(sdf$subj_idx), names(subj_traces))]
  
  # First sample non-regression variables 2/3...  
  betas <- rep(.5, n_samples) # z
  taus <- as.numeric(unlist(par_vals %>% select(contains("t_subj")) %>% 
                                select(-c((matches("v_")))) %>% sample_n(n_samples)))  
  alphas <- as.numeric(unlist(par_vals %>% select(contains("a_subj"))))
  
  # Get names of regression variables and sample rows from sdf w them 
  rpm <- c("val_ctr", "sess_ctr", 
           "valence", "session") # Store these just so it's easier to make sense of output
  coefs <- sdf[sort(sample(1:nrow(sdf), size=n_samples, replace=FALSE)), rpm]
  
  regr_traces <- par_vals %>% select((matches("v_"))) %>% sample_n(n_samples) 
  # Get rid of subject identifying info in colnames 
  colnames(regr_traces) <- unlist(map(strsplit(names(regr_traces), "_sub"), 1))
  
  # Construct regression variables 
  deltas <- FindDrifts1(coefs, regr_traces=regr_traces %>% select(contains("v_")), n_samples)
  
  pars <- list("alpha"=alphas, "betas"=betas, "taus"=taus, "deltas"=deltas)
  subj_sims <- lapply(1:n_samples, function(x) {
    # Generate n_sample RTs and responses with Rwiener package #
    # storing each along with the trial coefs for context #
    rwiener_outs <- setNames(
      data.frame(rwiener(1, alpha=alphas[x], tau=taus[x], beta=betas[x], delta = deltas[x])),
      c("rt", "response"))
    df_row <- data.frame(coefs[x, ], rwiener_outs, "ID"=unique(sdf$subj_idx))
    df_row 
  }) %>% bind_rows()
  
  subj_sims$response <- as.numeric(fct_relevel(subj_sims$response, "lower", "upper")) - 1
  
subj_sims
}
SimSubjBPlusNDT <- function(sdf, subj_traces, n_samples=100, ts=1, incl_z=0) {
  ### Same as SimSubjBM but adds non-decision time regressor ###
  n_samples <- min(nrow(sdf), n_samples)
  idx <- unique(sdf$subj_idx)
  if (length(idx) > 1) stop("Error in extracting subject pars: trying to pull for more than one subject.")
  
  par_vals <- subj_traces[grep(unique(sdf$subj_idx), names(subj_traces))]
  
  # First sample non-regression variables ...
  alphas <- as.numeric(unlist(par_vals %>% select(contains("a_subj"))))
  
  # Get names of regression variables and sample rows from sdf w them 
  rpm <- c("val_ctr", "sess_ctr", 
           "valence", "session") # Store these just so it's easier to make sense of output
  
  coefs <- sdf[sort(sample(1:nrow(sdf), size=n_samples, replace=FALSE)), rpm]
  if (!incl_z) {
    regr_traces <- par_vals %>% select((matches("v_|t_"))) %>% sample_n(n_samples) 
  } else {
    regr_traces <- par_vals %>% select((matches("v_|t_|z_"))) %>% sample_n(n_samples) 
  }
  # Get rid of subject identifying info in colnames 
  colnames(regr_traces) <- unlist(map(strsplit(names(regr_traces), "_sub"), 1))
  
  # Construct regression variables 
  deltas <- FindDrifts1(coefs, regr_traces=regr_traces %>% select(contains("v_")), n_samples)
  # Ndt 
  if (!ts) {
    taus <- FindNDT1(coefs, regr_traces=regr_traces %>% select(contains("t_")), n_samples) 
  } else {
    taus <- FindNDT2(coefs, regr_traces=regr_traces %>% select(contains("t_")), n_samples) 
  }
  if (!incl_z) {
    betas <- rep(.5, n_samples) # z
  } else {
    betas <- FindZ2(coefs, regr_traces=regr_traces %>% select(contains("z_")), n_samples) 
  }
  
  pars <- list("alpha"=alphas, "betas"=betas, "taus"=taus, "deltas"=deltas)
  subj_sims <- lapply(1:n_samples, function(x) {
    # Generate n_sample RTs and responses with Rwiener package #
    # storing each along with the trial coefs for context #
    rwiener_outs <- setNames(
      data.frame(rwiener(1, alpha=alphas[x], tau=taus[x], beta=betas[x], delta = deltas[x])),
      c("rt", "response"))
    df_row <- data.frame(coefs[x, ], rwiener_outs, "ID"=unique(sdf$subj_idx))
    df_row 
  }) %>% bind_rows()
  
  subj_sims$response <- as.numeric(fct_relevel(subj_sims$response, "lower", "upper")) - 1
  
  subj_sims
}
FindDrifts1 <- function(coefs, regr_traces, n_samples) {
  ### Finds effective drift each trial given the combination of coefficients #
  # present on the trial and trace samples of regression coefs ###
  pars <- 
    unlist(lapply(1:n_samples, function(x) {
      cr <- coefs[x, ]
      tr <- regr_traces[x, ]
      # Construct v this trial 
      # Start with intercept..
      v <- as.numeric(tr["v_Intercept"])
      # Main effects 
      v <- v + as.numeric(tr["v_sess_ctr"] * cr["sess_ctr"]) 
      v <- v + as.numeric(tr["v_val_ctr"] * cr["val_ctr"])
      # Two way interaction 
      v <- v + as.numeric(tr["v_val_ctr.sess_ctr"] * cr["sess_ctr"] * cr["val_ctr"]) 
    }))
  pars
}
FindNDT1 <- function(coefs, regr_traces, n_samples) {
  ### Finds effective drift each trial given the combination of coefficients #
  # present on the trial and trace samples of regression coefs ###
  pars <- 
    unlist(lapply(1:n_samples, function(x) {
      cr <- coefs[x, ]
      tr <- regr_traces[x, ]
      # Construct v this trial 
      # Start with intercept..
      t <- as.numeric(tr["t_Intercept"])
      # Main effects 
      t <- t + as.numeric(tr["t_val_ctr"] * cr["val_ctr"])
    }))
  
pars
}
FindNDT2 <- function(coefs, regr_traces, n_samples) {
  ### Finds effective drift each trial given the combination of coefficients #
  # present on the trial and trace samples of regression coefs ###
  pars <- 
    unlist(lapply(1:n_samples, function(x) {
      cr <- coefs[x, ]
      tr <- regr_traces[x, ]
      # Construct v this trial 
      # Start with intercept..
      t <- as.numeric(tr["t_Intercept"])
      # Main effects 
      t <- t + as.numeric(tr["t_val_ctr"] * cr["val_ctr"])
      t <- t + as.numeric(tr["t_sess_ctr"] * cr["sess_ctr"]) 
      # Two way interaction 
      t <- t + as.numeric(tr["t_val_ctr.sess_ctr"] * cr["sess_ctr"] * cr["val_ctr"]) 
    }))
  
  pars
}
### Transform the z-parameters to be bound between 0 and 1 ###
ScaleSP <- function(sp_trace) exp(sp_trace)/(1+exp(sp_trace))  
FindZ2 <- function(coefs, regr_traces, n_samples) {
  ### Finds effective drift each trial given the combination of coefficients #
  # present on the trial and trace samples of regression coefs ###
  pars <- 
    unlist(lapply(1:n_samples, function(x) {
      cr <- coefs[x, ]
      tr <- regr_traces[x, ]
      #browser()
      # Construct v this trial 
      # Start with intercept..
      
      z <- as.numeric(tr["z_Intercept"])
      # Main effects 
      z <- z + as.numeric(tr["z_val_ctr"] * cr["val_ctr"])
      z_on_correct_scale <- ScaleSP(z) 
      #print(z_on_correct_scale)
    }))
pars
}
# sdf <- bx_dfs %>% filter(subj_idx==30)
# subj_traces <- GetSubjTraces(tm)
# SimSubjBM(sdf, subj_traces)

