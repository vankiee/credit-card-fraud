library(ranger)
library(caret)
library(data.table)
library(caTools)
library(pROC)
library(rpart)
library(rpart.plot)
library(neuralnet)
library(gbm, quietly = TRUE)

credit_card_data <- read.csv("creditcard.csv")

# exploring the data
dim(credit_card_data)
head(credit_card_data, 6)

table(credit_card_data$Class)
summary(credit_card_data$Amount)
names(credit_card_data)
var(credit_card_data$Amount)
sd(credit_card_data$Amount)

# scaling the data
credit_card_data$Amount = scale(credit_card_data$Amount)
newData = credit_card_data[, -c(1)]

set.seed(123)
data_sample <- sample.split(newData$Class, SplitRatio = .80)
train_data <- subset(newData, data_sample == TRUE)
test_data = subset(newData, data_sample == FALSE)

# fitting logistic regression model
Logistic_Model = glm(Class~., test_data, family = binomial())
plot(Logistic_Model)

# ROC curve
lr.predict <- predict(Logistic_Model, train_data, probability = TRUE)
auc.gbm<- roc(test_data$Class, lr.predict, plot = TRUE, col = "blue")

# fit decision tree model
decisionTree_model <- rpart(Class~., credit_card_data, method = 'class')
predicted_val <- predict(decisionTree_model, credit_card_data, type = 'class')
probability <- predict(decisionTree_model, credit_card_data, type = 'prob')
rpart.plot(decisionTree_model)

# artificial neural network
ANN_model = neuralnet(Class~., train_data, linear.output = FALSE)
plot(ANN_model)

predANN = compute(ANN_model, test_data)
resultANN = predANN$net.result
resultANN = ifelse(resultANN > 0.5, 1, 0)

# gradient boosting
system.time(
    model_gbm <- gbm(Class ~ ., 
                    distribution = "bernoulli",
                    data = rbind(train_data, test_data), 
                    n.trees = 500, 
                    interaction.depth = 3, 
                    n.minobsinnode = 100, 
                    shrinkage = 0.01, 
                    bag.fraction = 0.5, 
                    train.fraction = nrow(train_data) /
                    (nrow(train_data)+ nrow(test_data))
                    )
)

# determine best iteration based on test data
gbm.iter = gbm.perf(model_gbm, method = "test")

model.influence = relative.influence(model_gbm, n.trees = gbm.iter, sort. = TRUE)
plot(model_gbm)

# AUC plot
gbm_test = predict(model_gbm, newdata = test_data, n.trees = gbm.iter)
gbm_auc = roc(test_data$Class, gbm_test, plot( = TRUE, col = "red")