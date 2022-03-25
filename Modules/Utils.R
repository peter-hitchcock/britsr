InvLogit <- function(x) exp(x) / (1 + exp(x))
HDDMnnInvLogit <- function(x) .1 + .9 * ( 1 / (1 + exp(-x) ))

HDDMnnInvLogit(-.4)
### For saving variables without overwriting ###
GenRandString <- function() paste0('_', toString(round(runif(1, 1000, 9999)))) 

SetPlotPars <- function() {
  ### Set up some plot aspects we'll reuse across functions ####
  
  # general aesthetics #
  ga <<- theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
               panel.background = element_blank(), axis.line = element_line(colour = "black"))
  # legend pars #
  lp <<- theme(legend.text = element_text(size = 20),
               legend.title = element_blank(),
               legend.key.size = unit(2.5, 'lines'))
  
  lp_multi <<- theme(legend.text = element_text(size = 15),
                     legend.title = element_blank(),
                     legend.key.size = unit(1.25, 'lines'),
                     legend.position = c(.9, 1.15))
  
  lp_ppc <<- theme(legend.text = element_text(size = 15),
                   legend.title = element_blank(),
                   legend.key.size = unit(1.25, 'lines'),
                   legend.position = c(.9, .9))
  
  # axis pars #
  ap <<- theme(axis.text = element_text(size=35),
               axis.title = element_text(size=35))
  
  tol <<- theme(legend.position = "none") 
  
  ft <<- theme(strip.text=element_text(size = 30)) 
  
  # title pars #
  tp <<- theme(plot.title = element_text(size = 20, face='bold', hjust = .5))
  
  # tp_multi <<- theme(plot.title = element_text(margin=margin(b=-8), 
  #                                              size=15, face='bold', hjust=.05, vjust=.9)) 
  # color pars #
  cf_vals <<- c('yellow', 'purple')
} 

### Returns just the group-level or subject-level traces ###
GetGrpTraces <- function(x) x %>% select(-c(contains("_subj")))
GetSubjTraces <- function(x) x %>% select(contains("_subj"))