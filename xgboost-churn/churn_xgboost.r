library(tidyverse)
library(caret)
library(xgboost)
library(Ckmeans.1d.dp)
library(DiagrammeR)
library(precrec)
library(SHAPforxgboost)
options(warn = -1)

data <- read.csv("data/churn.csv")
head(data)

# Check dimension

# Transformations: 
# 1) Remove unnecesary vars
# 2) Convert to numeric

data <- data %>%
   select(-RowNumber, -CustomerId, -Surname) 
head(data)

dummy_obj <- dummyVars(~Geography + Gender, data)
dummy_df <- predict(dummy_obj, newdata = data)
data <- cbind(select(data, -Geography, -Gender), dummy_df)
head(data)

options(repr.plot.width=12, repr.plot.height=4)

for (i in 1:ncol(data)){
    p <- ggplot(data)
    p <- p + geom_histogram(aes(x=data[,i], y=..density.., fill=factor(Exited)), alpha = 0.2)
    p <- p + geom_density(aes(x=data[,i], y =..density.., fill = factor(Exited), colour = factor(Exited)), 
                          alpha = 0.35)
    p <- p + scale_x_continuous(name = names(data)[i])
    p <- p + theme_minimal()
    print(p)
}

# See outliers
for (i in 1:ncol(data)){
    p <- ggplot(data)
    p <- p + geom_boxplot(aes(x=factor(Exited), y=data[,i], fill=factor(Exited)), alpha = 0.2)
    p <- p + scale_y_continuous(name = names(data)[i])
    p <- p + theme_minimal()
    print(p)
}

# Data partition
set.seed(42)

train_index <- createDataPartition(data$Exited, p = .7, list = FALSE, times = 1)

# First partition: 70% train - 30% test 
train <- data[train_index, ]  
test <- data[-train_index, ]

round(table(train$Exited)/nrow(train)*100, 2)
round(table(test$Exited)/nrow(test)*100, 2)

head(train)

class(train)

# Predictive variables in training dataset
X_train <- train %>%
    select(-Exited) %>%
    data.matrix()
# Labels in training dataset
y_train <- train$Exited

X_test <- test %>%
    select(-Exited) %>%
    data.matrix()
y_test <- test$Exited

set.seed(42)
xgb <- xgboost(data = X_train, 
 label = y_train, 
 eta = 0.2,
 max_depth = 3, 
 nround = 10, 
 subsample = 0.5,
 colsample_bytree = 0.5,
 seed = 1,
 eval_metric = "auc",
 objective = "binary:logistic",
 nthread = 3,
 scale_pos_weight = 4
)

# This info is accesible
xgb$evaluation_log

pred <- predict(xgb, X_test)
head(pred)

head(pred > 0.5)

cbind(pred > 0.5, y_test) %>% 
  data.frame() %>% 
  table() %>% 
  confusionMatrix(positive = "1")    # from caret package again

options(repr.plot.width=8, repr.plot.height=6.5)

precrec_obj <- evalmod(scores = pred, labels = y_test)
autoplot(precrec_obj)

feature_importance <- xgb.importance(feature_names = xgb$feature_names, model = xgb)
feature_importance

options(repr.plot.width = 8, repr.plot.height = 8)
xgb.ggplot.importance(importance_matrix = feature_importance, rel_to_first = TRUE)

xgb.plot.tree(feature_names = xgb$feature_names, model = xgb, trees = 0)

# To prepare the long-format data:
shap_long <- shap.prep(xgb_model = xgb, X_train = X_train)

options(repr.plot.width = 12, repr.plot.height = 10)
shap.plot.summary(shap_long)
