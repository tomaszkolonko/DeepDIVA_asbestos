# Create data:

setwd("/Users/tomasz/DeepDIVA_asbestos/MasterThesisLatex/extra_stuff/cnn_baseline/cnn_basic")

train_csv <- "cnn_train.csv"
train_csv_pre <- "resnet18-pre-train-accuracy.csv"
val_csv <- "cnn_val.csv"
val_csv_pre <- "resnet18-pre-val-accuracy.csv"

title_train <- "Training Accuracy for CNN_Basic"
title_val <- "Validation Accuracy for CNN_Basic"

lr_decay_rate <- 20
op <- par(cex = 1.1)

# #####################################################################################################
# ##########################   TRAINING ACCURACY ON CLASSIFICATION    #################################
# #####################################################################################################

style_train <- read.csv(train_csv, header=TRUE, sep=",")
style_train_pre <- read.csv(train_csv_pre, header=TRUE, sep=",")
style_val <- read.csv(val_csv, header=TRUE, sep=",")
style_val_pre <- read.csv(val_csv_pre, header=TRUE, sep=",")

# Make a basic graph for Training Accuracy
plot(style_train$Value ~ style_train$Step, type="l", bty="l", xlab="Step", ylab="Accuracy",
     col=rgb(0.2,0.4,0.1,0.7), lwd=3, pch=17, ylim=c(0,100), main=title_train)
lines(style_train_pre$Value ~ style_train_pre$Step, col=rgb(0.8,0.4,0.1,0.7),
      lwd=3, pch=19, type="l")

# Make vertical lines to display lr-decay
abline(v=lr_decay_rate, col="grey")
abline(v=2*lr_decay_rate, col="grey")

# Add a legend
op <- par(cex = 1.1)
legend("bottomright", 
       legend = c("From scratch"), 
       col = c(
               rgb(0.2,0.4,0.1,0.7)), 
       pch = c(19,19), 
       bty = "n", 
       pt.cex = 1, 
       cex = 1.2, 
       text.col = "black", 
       horiz = F , 
       inset = c(0, 0))




# #####################################################################################################
# ######################   VALIDATION ACCURACY ON CLASSIFICATION    ###################################
# #####################################################################################################


# Make a basic graph on Validation Accuracy
plot(style_val$Value ~ style_val$Step, type="l", bty="l", xlab="Step", ylab="Accuracy",
     col=rgb(0.2,0.4,0.1,0.7), lwd=3, pch=17, ylim=c(0,100), main=title_val)
lines(style_val_pre$Value ~ style_val_pre$Step, col=rgb(0.8,0.4,0.1,0.7),
      lwd=3, pch=19, type="l")

# Make vertical lines to display lr-decay
abline(v=lr_decay_rate, col="grey")
abline(v=2*lr_decay_rate, col="grey")

# Add a legend
op <- par(cex = 1.1)
legend("bottomright", 
       legend = c("From scratch"), 
       col = c(
               rgb(0.2,0.4,0.1,0.7)), 
       pch = c(19,19), 
       bty = "n", 
       pt.cex = 1, 
       cex = 1.2, 
       text.col = "black", 
       horiz = F , 
       inset = c(0, 0))












install.packages('caret', dependencies = TRUE)
library(caret)

# construct the evaluation dataset
set.seed(144)
true_class <- factor(sample(paste0("Class", 1:2), size = 1000, prob = c(.2, .8), replace = TRUE))
true_class <- sort(true_class)
class1_probs <- rbeta(sum(true_class == "Class1"), 4, 1)
class2_probs <- rbeta(sum(true_class == "Class2"), 1, 2.5)
test_set <- data.frame(obs = true_class,Class1 = c(class1_probs, class2_probs))
test_set$Class2 <- 1 - test_set$Class1
test_set$pred <- factor(ifelse(test_set$Class1 >= .5, "Class1", "Class2"))

cm <- confusionMatrix(data = test_set$pred, reference = test_set$obs)

draw_confusion_matrix <- function(cm) {
  
  
  res <- c(100,21,36,144)
  total = res[1] + res[2] + res[3] + res[4]
  
  cm_data <- vector(mode="list", length=3)
  names(cm_data) <- c("Accuracy", "Recall", "Precision")
  
  # Generate color gradients. Palettes come from RColorBrewer.
  greenPalette <- c("#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B")
  redPalette <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A","#EF3B2C","#CB181D","#A50F15","#67000D")
  getColor <- function (greenOrRed = "green", amount = 0) {
    if (amount == 0)
      return("#FFFFFF")
    palette <- greenPalette
    if (greenOrRed == "red")
      palette <- redPalette
    colorRampPalette(palette)(100)[10 + ceiling(90 * amount / total)]
  }
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col=getColor("green", res[1]))
  text(195, 435, "asbestos", cex=1.2)
  rect(250, 430, 340, 370, col=getColor("red", res[3]))
  text(295, 435, "non-asbestos", cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col=getColor("red", res[2]))
  rect(250, 305, 340, 365, col=getColor("green", res[4]))
  text(140, 400, "asbestos", cex=1.2, srt=90)
  text(140, 335, "non-asbestos", cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')

  recall = res[1] / res[1] + res[3]
  precision = res[1] / res[1] + res[2]
  
  cm_data[[1]] = res[1] + res[4] / total
  cm_data[[2]] = recall
  cm_data[[3]] = precision
  
  actual_true = res[1] + res[2]
  actual_false = res[3] + res[4]
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm_data[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm_data[[1]]), 3), cex=1.2)
  
  
  text(50, 85, names(cm_data[2]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm_data[[2]]), 3), cex=1.2)

  text(90, 85, names(cm_data[3]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm_data[[3]]), 3), cex=1.2)
}

draw_confusion_matrix(cm)

