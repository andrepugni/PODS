pacman::p_load(rddensity, lpdensity)
library(stringr)
library(ggplot2)
path <- "resultsRAW/"
dataset <- "galaxyzoo"
df.cal <- read.csv(paste0(path, dataset, "/GCresultsRAW_cal_galaxyzoo_CC_42_ep50.csv"))
df.tst <- read.csv(paste0(path, dataset, "/GCresultsRAW_test_galaxyzoo_CC_42_ep50.csv"))

cc_col = rgb(0.00392156862745098, 0.45098039215686275, 0.6980392156862745)
rs_col = rgb(0.8705882352941177, 0.5607843137254902, 0.0196078431372549)
asm_col = rgb(0.00784313725490196, 0.6196078431372549, 0.45098039215686275)
image_fold = "figs/"


for (x in c(.1,.2,.3,.4,.5,.6,.7,.8,.9)) {
  cutoff <- quantile(df.cal$rej_score, probs = x)
  
  res <- rddensity(df.tst$rej_score, c = cutoff)
  summary(res)
  pv.perm <- format(round(res$bino$pval[length(res$bino$pval)], digits = 5), scientific = TRUE)
  plot_title = paste0(dataset," - ", str_replace(paste0(x,"0"),"0.","."))
  path_plot = paste0(image_fold,"/density_",str_replace(plot_title," ","_"),".png")
  rdp <- rdplotdensity(res, df.tst$rej_score, lcol = c("black", "black"), 
                       title = plot_title,
                       ylabel = "Reject Score Density",xlabel = "Reject Score", 
                       CItype = "none",lwd=c(1.1,1.1), histFillShade = .60,
                       histFillCol = cc_col)
  rdp$Estplot+ theme(plot.title = element_text(hjust=.5, size=28), 
                     axis.text = element_text(size=18),
                     axis.title = element_text(size=24)
  )+geom_text(
    aes(x = Inf, y = +Inf , label = paste0("p-value: ", pv.perm )), size=10,
    vjust = "inward", hjust = "inward") 
  ggsave(path_plot, width=2361, height=1836, units =("px"))
  
  
  
}

dataset <- "cifar10h"
df.cal <- read.csv(paste0(path, dataset, "/GCresultsRAW_cal_cifar10h_CC_42_ep150.csv"))
df.tst <- read.csv(paste0(path, dataset, "/GCresultsRAW_test_cifar10h_CC_42_ep150.csv"))

for (x in c(.1,.2,.3,.4,.5,.6,.7,.8,.9)) {
  cutoff <- quantile(df.cal$rej_score, probs = x)
  
  res <- rddensity(df.tst$rej_score, c = cutoff)
  summary(res)
  pv.perm <- format(round(res$bino$pval[length(res$bino$pval)], digits = 3), scientific = TRUE)
  plot_title = paste0(dataset," - ", str_replace(paste0(x,"0"),"0.","."))
  path_plot = paste0(image_fold,"/density_",str_replace(plot_title," ","_"),".png")
  rdp <- rdplotdensity(res, df.tst$rej_score, lcol = c("black", "black"), 
                       title = plot_title,
                       ylabel = "Reject Score Density",xlabel = "Reject Score", 
                       CItype = "none",lwd=c(1.1,1.1), histFillShade = .60,
                       histFillCol = cc_col)
  rdp$Estplot+ theme(plot.title = element_text(hjust=.5, size=28), 
                     axis.text = element_text(size=18),
                     axis.title = element_text(size=24)
  )+geom_text(
    aes(x = Inf, y = +Inf , label = paste0("p-value: ", pv.perm )), size=10,
    vjust = "inward", hjust = "inward") 
  ggsave(path_plot, width=2361, height=1836, units =("px"))
  
  
  
}



dataset <- "chestxray2"
df.cal <- read.csv(paste0(path, dataset, "/GCresultsRAW_cal_chestxray2_RS_42_ep3.csv"))
df.tst <- read.csv(paste0(path, dataset, "/GCresultsRAW_test_chestxray2_RS_42_ep3.csv"))

cutoff <- quantile(df.cal$rej_score, probs = 0.3)

res <- rddensity(df.tst$rej_score, c = cutoff)
summary(res)
rdplotdensity(res, df.tst$rej_score, lcol = c("black", "black"), xlabel = "Reject Score",
              histFillCol=rs_col, histFillShade = 0.6, CItype = "none")
dataset <- "xray-airspace"
for (x in c(.1,.2,.3,.4,.5,.6,.7,.8,.9)) {
  cutoff <- quantile(df.cal$rej_score, probs = x)
  
  res <- rddensity(df.tst$rej_score, c = cutoff)
  summary(res)
  pv.perm <- format(round(res$bino$pval[length(res$bino$pval)], digits = 3), scientific = TRUE)
  plot_title = str_replace(paste0(dataset," - ", x,"0"),"0.",".")
  path_plot = paste0(image_fold,"/density_",str_replace(plot_title," ","_"),".png")
  rdp <- rdplotdensity(res, df.tst$rej_score, lcol = c("black", "black"), 
                       title = plot_title,
                       ylabel = "Reject Score Density",xlabel = "Reject Score", 
                       CItype = "none",lwd=c(1.1,1.1), histFillShade = .60,
                       histFillCol = rs_col)
  rdp$Estplot+ theme(plot.title = element_text(hjust=.5, size=28), 
                     axis.text = element_text(size=18),
                     axis.title = element_text(size=24)
  )+geom_text(
    aes(x = Inf, y = +Inf , label = paste0("p-value: ", pv.perm )), size=10,
    vjust = "inward", hjust = "inward") 
  ggsave(path_plot, width=2361, height=1836, units =("px"))
  
  
  
}


dataset <- "hatespeech"
df.cal <- read.csv(paste0(path, dataset, "/GCresultsRAW_cal_hatespeech_RS_42_ep100.csv"))
df.tst <- read.csv(paste0(path, dataset, "/GCresultsRAW_test_hatespeech_RS_42_ep100.csv"))

for (x in c(.1,.2,.3,.4,.5,.6,.7,.8,.9)) {
  cutoff <- quantile(df.cal$rej_score, probs = x)
  
  res <- rddensity(df.tst$rej_score, c = cutoff)
  summary(res)
  pv.perm <- format(round(res$bino$pval[length(res$bino$pval)], digits = 3), scientific = TRUE)
  plot_title = str_replace(paste0(dataset," - ", x,"0"),"0.",".")
  path_plot = paste0(image_fold,"/density_",str_replace(plot_title," ","_"),".png")
  rdp <- rdplotdensity(res, df.tst$rej_score, lcol = c("black", "black"), 
                       title = plot_title,
                       ylabel = "Reject Score Density",xlabel = "Reject Score", 
                       CItype = "none",lwd=c(1.1,1.1), histFillShade = .60,
                       histFillCol = rs_col)
  rdp$Estplot+ theme(plot.title = element_text(hjust=.5, size=28), 
                     axis.text = element_text(size=18),
                     axis.title = element_text(size=24)
  )+geom_text(
    aes(x = Inf, y = +Inf , label = paste0("p-value: ", pv.perm )), size=10,
    vjust = "inward", hjust = "inward") 
  ggsave(path_plot, width=2361, height=1836, units =("px"))
  
  
  
}

dataset <- "synth"
df.cal <- read.csv(paste0(path, dataset, "/GCresultsRAW_cal_synth_ASM_42_0.1_0.4_0.1_ep50.csv"))
df.tst <- read.csv(paste0(path, dataset, "/GCresultsRAW_test_synth_ASM_42_0.1_0.4_0.1_ep50.csv"))

for (x in c(.1,.2,.3,.4,.5,.6,.7,.8,.9)) {
  cutoff <- quantile(df.cal$rej_score, probs = x)
  
  res <- rddensity(df.tst$rej_score, c = cutoff)
  summary(res)
  pv.perm <- format(round(res$bino$pval[length(res$bino$pval)], digits = 3), scientific = TRUE)
  plot_title = str_replace(paste0(dataset," - ", x,"0"),"0.",".")
  path_plot = paste0(image_fold,"/density_",str_replace(plot_title," ","_"),".png")
  rdp <- rdplotdensity(res, df.tst$rej_score, lcol = c("black", "black"), legendTitle = c("below cutoff", "above cutoff"),
                       title = plot_title,
                       ylabel = "Reject Score Density",xlabel = "Reject Score", 
                       CItype = "none",lwd=c(1.1,1.1), histFillShade = .60,
                       histFillCol = asm_col)
  rdp$Estplot+ theme(plot.title = element_text(hjust=.5, size=28), 
                     axis.text = element_text(size=18),
                     axis.title = element_text(size=24)
  )+geom_text(
    aes(x = Inf, y = +Inf , label = paste0("p-value: ", pv.perm )), size=10,
    vjust = "inward", hjust = "inward") 
  ggsave(path_plot, width=2361, height=1836, units =("px"))
  
  
  
}


