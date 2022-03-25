unique_items <- unique(d1_lf$item)
d1m_s <- d1m %>% filter(ID %in% d1_lf$ID)
all_res <- foreach (i = 1:length(unique_items)) %do% {
  browser()
  this_item <- d1_lf %>% filter(item==unique_items[i])
  d1m_s_tmp <- d1m_s %>% filter(ID %in% unique(this_item$ID))
  #print(this_item)
  if (all(this_item$ID==d1m_s_tmp$ID)) {
    res <- cor.test(this_item$scores, d1m_s_tmp$v_val_ctr_)
    out <- data.table("item"=unique(this_item$item),
                      "r"=as.numeric(res$estimate),
                      "ci95_low"=as.numeric(res$conf.int[1]),
                      "ci95_high"=as.numeric(res$conf.int[2]),
                      "p"=as.numeric(res$p.value))
  } else {
    
  }
  out
} %>% bind_rows()