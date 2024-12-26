# Init ####
setwd(sprintf("%s/..",dirname(rstudioapi::getActiveDocumentContext()$path)))
len = length

# define functions
init_f1_scores <- function() {
  # no supp data
  f1_cb = c(0.291685495322798,
            0.290538524646847,
            0.290412255148650,
            0.288536066465375,
            0.287590554814753)
  
  f1_svd= c(0.312886178855732,
            0.311193307593835,
            0.313071562396729,
            0.309036929318276,
            0.309975311906867)
  
  f1_ae = c(0.423586065267642,
            0.421824257395915,
            0.421817913165186,
            0.419740361120181,
            0.421213272064187)
  
  f1_dae = c(0.418663547501399,
             0.417099524430693,
             0.411829119780360,
             0.414655878846822,
             0.410250864972675)
  
  f1_vae = c(0.272965500464583,
             0.271945665059245,
             0.271693722160349,
             0.270636166978677,
             0.270825164257100)
  
  f1_aae = c(0.423268627226411,
             0.422001573096629,
             0.419841048644308,
             0.422145880954489,
             0.422211592280500)
  
  # yes sup data
  
  f1_sae = c(0.430532346051110,
             0.425333867965878,
             0.427641109519037,
             0.424362141442481,
             0.426472453098458)
  
  f1_sdae = c(0.414545584369016,
              0.415568804482477,
              0.415019177155389,
              0.417295656908460,
              0.413359172377368)
  
  f1_svae = c(0.283511846040660,
              0.282927304769428,
              0.282965733266931,
              0.280917596831154,
              0.280900111681369)
  
  f1_saae = c(0.430445769106625,
              0.426542632890866,
              0.424677148290504,
              0.425287704761856,
              0.425155176897973)
  
  
  f1_scores = data.frame(
    cb = f1_cb,
    svd = f1_svd,
    ae = f1_ae,
    dae = f1_dae,
    vae = f1_vae,
    aae = f1_aae,
    sae = f1_sae,
    sdae = f1_sdae,
    svae = f1_svae,
    saae = f1_saae
  )
  return(f1_scores)
}

init_map_scores <- function() {
  map_cb = c(0.218496175119998, 0.217299373527110, 0.217326167381779, 0.215459410281809, 0.214299447862093)
  
  map_svd= c(0.245588703233853, 0.244334659539414, 0.246525674642377, 0.242177282776786, 0.242181644119886)
  
  map_ae = c(0.388831014366862, 0.386997484005011, 0.387988907364513, 0.383682072740080, 0.384647979338682)
  
  map_dae = c(0.380019019359311, 0.377992866153813, 0.372156956464703, 0.373130094317923, 0.367915519373991)
  
  map_vae = c(0.197645537340875, 0.196030510963199, 0.196140608895228, 0.195509885752387, 0.195317723053603)
  
  map_aae = c(0.387341668352564, 0.386111669444762, 0.384745231083297, 0.386333110832434, 0.386005557041575)
  
  map_sae = c(0.397415490288867, 0.390863441813550, 0.395728282967386, 0.389454048986587, 0.391553867454534)
  
  map_sdae = c(0.373764002359287, 0.376264075840449, 0.375331061933583, 0.377763021575841, 0.371919285344189)
  
  map_svae = c(0.209554892535574, 0.208760984428560, 0.209385135904287, 0.207209074827483, 0.207085389020440)
  
  map_saae = c(0.396899925664454, 0.392705084368235, 0.389685555597399, 0.390721568000891, 0.388476418713609)
  
  map_scores = data.frame(
    cb = map_cb,
    svd = map_svd,
    ae = map_ae,
    dae = map_dae,
    vae = map_vae,
    aae = map_aae,
    sae = map_sae,
    sdae = map_sdae,
    svae = map_svae,
    saae = map_saae
  )
  return(map_scores)
}

perform_robust_test_diff_perf <- function(metric_vals, test_type="sign") {
  # `metric_vals` should be a data frame or matrix where:
  # - Rows represent folds
  # - Columns represent models
  model_names = colnames(metric_vals)
  n_models = ncol(metric_vals)
  results = list()
  
  # Perform pairwise Wilcoxon signed-rank tests
  for (i in 1:n_models) {
    #for (j in (i + 1):n_models) {
    for (j in 1:n_models) {
      if(i == j)
        next
      model_1 = model_names[i]
      model_2 = model_names[j]
      if(test_type == "wilcox") {
        # Perform the Wilcoxon signed-rank test
        test_result = wilcox.test(metric_vals[, i], metric_vals[, j], paired = TRUE, alternative = "two.sided")
        
        # Store the result in a readable format
        results[[paste(model_1, "vs", model_2)]] = list(
          p_value = test_result$p.value,
          statistic = test_result$statistic,
          method = test_result$method,
          input_metric_vals = list(model_1 = metric_vals[, i], model_2 = metric_vals[, j])
        )
      }
      if(test_type == "sign") {
        # Perform the Sign Test
        test_result = binom.test(
          sum(metric_vals[, i] > metric_vals[, j]), # Number of folds where Model i outperforms Model j
          nrow(metric_vals),                     # Total number of folds
          p = 0.5,                             # Null hypothesis: no preference between models (equal probability)
          alternative = "two.sided"            # Two-sided test
        )
        
        # Store the result in a readable format
        results[[paste(model_1, "vs", model_2)]] = list(
          p_value = test_result$p.value,
          statistic = test_result$statistic,
          method = test_result$method,
          input_metric_vals = list(model_1 = metric_vals[, i], model_2 = metric_vals[, j])
        )
      }
      
    }
  }
  p_values = sapply(results, function(x) x$p_value)
  p_values_adj = p.adjust(p_values, method = "bonferroni")
  for(i in 1:len(results)) {
    k = names(results)[i]
    v = results[[k]]
    v$p_value_adj = p_values_adj[i]
    results[[k]] = v
  }
  
  # Return results as a named list
  return(results)
}

make_df_for_heatmap <- function(sign_test_res, scores) {
  hm_df = NULL
  for(model_nm in names(scores))
  {
    rel_comparison_names = sapply(1:len(sign_test_res), function(i) if(startsWith(names(sign_test_res)[i], sprintf("%s vs ", model_nm))) names(sign_test_res)[i] else NULL )
    rel_comparison_names[sapply(rel_comparison_names, is.null)] = NULL # remove nulls
    rel_comparisons = sapply(1:len(sign_test_res), function(i) if(startsWith(names(sign_test_res)[i], sprintf("%s vs ", model_nm))) sign_test_res[[i]] else NULL )
    rel_comparisons[sapply(rel_comparisons, is.null)] = NULL # remove nulls
    
    names(rel_comparisons) = rel_comparison_names
    
    
    n_counts = sapply(rel_comparisons, function(x) x$statistic)
    c_row = as.data.frame(list(model=model_nm))
    for(other_model_nm in names(scores)) {
      if(other_model_nm == model_nm)
        c_row[, sprintf("count_better_than_%s", other_model_nm)] = 0
      else
        c_row[, sprintf("count_better_than_%s", other_model_nm)] = n_counts[sprintf('%s vs %s.number of successes', model_nm, other_model_nm)]
    }
    hm_df = rbind(hm_df, c_row)
  }
  
  rm(c_row, sign_test_res, scores, rel_comparison_names, rel_comparisons)
  rm(model_nm, n_counts, other_model_nm)
  
  
  model_names = hm_df$model
  model_names_pretty = c( "CB" ,"SVD" ,"AE" ,"DAE" ,"VAE" ,"AAE" ,"AE" ,"DAE" ,"VAE" ,"AAE")
  hm_df$model = NULL
  colnames(hm_df) = model_names_pretty
  rownames(hm_df) = seq(1, nrow(hm_df), 1)
  return(hm_df)
}

plot_sign_test_heatmap <- function(hm_df, model_types) {
  model_names = colnames(hm_df)
  Heatmap(
    hm_df,
    col = circlize::colorRamp2(c(0,2,3,5), c("#0628bf", "#faf2ac", "#e8faac",  "#e30505") ),
    name = "Sign Test",
    #top_annotation = column_ha,
    #left_annotation = row_ha,
    column_split = model_types,
    row_split = model_types,
    show_row_dend = F,
    show_column_dend = F,
    heatmap_legend_param = list(title = "Sign-test count", at = seq(0,5,1)),
    column_title_gp = gpar(fontsize=16),
    row_title_gp = gpar(fontsize=16),
    row_names_gp = gpar(fontsize=12),
    cell_fun = function(j,i,x,y,width,height,fill) {
      grid.rect(x=x, y=y, width=width, height=height,
                gp=gpar(col='grey', fill=NA))
      if(i==j) {
        grid.rect(x=x, y=y, width=width, height=height,
                  gp=gpar(col='grey', fill='#9e9e9e'))
      }
    },
    cluster_rows = F,
    cluster_columns = F,
    row_labels = model_names,
    border=T
  )
  
}



# create/prep data for plotting
model_types = factor(c(rep("Medication only", 6), rep("Medication +\n patient data", 4)))
f1_scores = init_f1_scores()
f1_res = perform_robust_test_diff_perf(f1_scores, 'sign')
result_df_f1 = make_df_for_heatmap(f1_res, f1_scores)



map_scores = init_map_scores()
map_res = perform_robust_test_diff_perf(map_scores, 'sign')
result_df_map = make_df_for_heatmap(map_res, map_scores)


# Make plots ####
library(devtools)
library(ComplexHeatmap)

plot_sign_test_heatmap(result_df_f1, model_types)

plot_sign_test_heatmap(result_df_map, model_types)
