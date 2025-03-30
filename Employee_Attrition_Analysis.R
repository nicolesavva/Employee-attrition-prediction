# ************************************************
# R Script For Employee Attrition Analysis

#  clears all objects in "global environment"
rm(list=ls())

# ************************************************
# Global Environment variables

DATASET_FILENAME  <- "Employee_Attrition_Analysis.csv" # Name of input dataset file

# ************************************************

# Libraries used for this project are taken from CRAN

# ************************************************

MYLIBRARIES<-c("outliers",
               "corrplot",
               "MASS",
               "formattable",
               "stats",
               "ggplot2",
               "tidyr",
               "reshape2",
               "caret",
               "webr",
               "rpart",
               "randomForest",
               "pROC",                   # library used for ROC
               "smotefamily",            # library used for SMOTE 
               "vcd",
               "class",                  # library for implementing KNN
               "e1071",                  # library for implementing SVM  
               "pander",                 # library used for presenting results in a table
               "knitr",
               "keras",                    # library for Neural Networks
               "PerformanceAnalytics")

# ************************************************
# User defined functions are next
# ************************************************

# ************************************************
# borderPrint():
#
# This function prints the provided text with a decorative border of a specified
# character and width.
#
# INPUT: text - The text to be printed.
#        char - The character used to create the border (default is "*").
#        width - The width of the border (default is 30).
#
# OUTPUT: Prints the text with the border
#
# ************************************************
borderPrint <- function(text, char = "*", width = 30) {
  cat(rep(char, width), "\n")
  cat(text, "\n")
  cat(rep(char, width), "\n")
}

# ************************************************

# smote_balancing(dataset, target_feature):
#
# This function balances the imbalanced data
#
# INPUT: dataset - The dataframe object to be balanced.
#        target_feature - target feature
#
# OUTPUT: This function returns balanced dataframe object

# ************************************************
smote_balancing <- function(dataset, target_feature) {
  
  X_features <- dataset[, -which(colnames(dataset) == target_feature)]
  target <- dataset[[target_feature]]
  
  smote_results <- SMOTE(X = X_features,
                        target = target,      # target class attribute
                        K = 5,                # number of nearest neighbors during sampling process 
                        dup_size = 4.202532)  # desired times of synthetic minority instances over the original number of majority instances
  
  oversampled_data = smote_results$data
  # Changing the "class" name to "Attrition"
  names(oversampled_data)[which(names(oversampled_data) == "class")] <- "Attrition"
  
  return(oversampled_data)
}

# ************************************************

# ************************************************

# checkMissingValues(dataset):
#
# This function checks whether any missing values are present in
# a provided dataset and returns those missing values
#
# INPUT: dataset - The dataset for which the missing value are to be checked
#
# OUTPUT: This function returns missing values in a dataset.
# ************************************************

checkMissingValues <- function(dataset){
  
  missing_values <- colSums(is.na(dataset)) > 0
  print(missing_values)
  print(typeof(missing_values))
  
  # Columns containing missing values
  columns_with_missing_values <- names(missing_values[missing_values])
  print(columns_with_missing_values)
  
  if (length(columns_with_missing_values) == 0) {
    # No missing values are present
    return(NULL)
  } else {
    # Return the missing values found
    missing_values <- dataset[, columns_with_missing_values]
    return(missing_values)
  }
}

# ************************************************

# checkForDuplicates(dataset):
#
# This function checks whether any duplicates are present in the 
# dataset and returns those duplicates
#
# INPUT: dataset - The dataset for which the duplicates are to be checked
#
# OUTPUT: This function returns duplicates in a dataset.
# ************************************************

checkForDuplicates <- function(dataset){
  duplicates <- duplicated(dataset)
  if (any(duplicates)) {
    print("Duplicate rows found:")
    return(duplicates)
  } else {
    print("No duplicate rows found.")
    return(NULL)
  }
}

# ************************************************
# MinMaxScaling(dataset) :
#
# Perform MinMax Scaling of the dataset

# INPUT: dataframe - a dataframe object
#
# OUTPUT: MinMax Scaled dataframe
# ************************************************
MinMaxScaling <- function(dataset) {
  
  # scaling the dataframe
  preProcessing <- preProcess(dataset, method = c("range"))
  
  # Transform the data
  # MinMaxScaled <- predict(MinMaxScaling, combined_data)
  MinMaxScaled <- predict(preProcessing, dataset)
  
  return(MinMaxScaled)
}

# ************************************************
# plot_two_features_against_attrition_stacked(data, feature1, feature2) :
#
# Plot two features together from the dataset

# INPUT: data - a dataframe object
#        feature1 - Feature 1 for plotting
#        feature2 - Feature 2 for plotting
#
# OUTPUT: plotting two features together
# ************************************************

# Function for plotting two or more features against Attrition

plot_two_features_against_attrition_stacked <- function(data, feature1, feature2) {
  if (feature1 == "Attrition" || feature2 == "Attrition") {
    cat("Cannot plot 'Attrition' against itself.\n")
    return()
  }
  
  if (!feature1 %in% colnames(data) || !feature2 %in% colnames(data)) {
    cat("Feature(s) not found in the data.\n")
    return()
  }
  
  # Create a new combined feature
  data$combined_feature <- with(data, paste(as.factor(data[[feature1]]), as.factor(data[[feature2]]), sep = " - "))
  
  # Plot the combined feature against Attrition
  p <- ggplot(data, aes(x = combined_feature, fill = Attrition)) +
    geom_bar(position = "stack") +
    labs(title = paste("Attrition vs", feature1, "and", feature2), x = paste(feature1, "/", feature2), y = "Count") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability
  
  # Add count labels
  p <- p + geom_text(stat = "count", aes(label = ..count..), position = position_stack(vjust = 0.5), color = "white")
  return(p)
}

# ************************************************
# createChart(data, feature_column) :
#
# Plot a pie chart of individual features from the dataset

# INPUT: data - a dataframe object
#        feature_column - feature column name
#
# OUTPUT : chart - pie chart of the feature name provided
# ************************************************
# Function to create a donut chart from a feature column
createChart <- function(data, feature_column) {
  # Get the unique counts of the feature column
  value_counts <- table(data[[feature_column]])
  
  # Create a data frame from value_counts
  value_counts_df <- as.data.frame(value_counts)
  
  # Create a donut chart
  feature_chart <- ggplot(value_counts_df, aes(x = "", y = Freq, fill = factor(Var1))) +
    geom_bar(stat = "identity", width = 1) +
    coord_polar("y", start = 0) +
    labs(title = paste(feature_column, "Chart")) +
    scale_fill_discrete(name = feature_column) +
    geom_text(aes(label = paste(Freq)), position = position_stack(vjust = 0.5)) + 
    theme_void()
  plot(feature_chart)
}

# ************************************************
# plotTwoFeatures(data, feature1, feature2) :
#
# Create a bar plot of two features from the dataset

# INPUT:  data - dataframe object
#         feature1 - feature 1 to plot
#         feature2 - feature 2 to plot
#
# OUTPUT : chart - pie chart of the features provided
# ************************************************
# Function to create a bar plot from two feature columns

plotTwoFeatures <- function(data, feature1, feature2) {
  plots <- lapply(feature1, function(feat) {
    dual_plot <- ggplot(data, aes(x = .data[[feat]], fill = .data[[feature2]])) +
      geom_bar(position = "dodge") +
      geom_text(stat = "count", aes(label = after_stat(count), vjust = -0.7), position = position_dodge(0.9)) +
      labs(
        title = paste("Distribution of", feature2, "by", feat),
        x = feat,
        y = "Count"
      ) +
      scale_fill_discrete(name = feature2) +
      theme_minimal()
    
    print(dual_plot)  # Print each plot as it's generated
  })
  return(plots)
}
# ************************************************
# THIS FUNCTION HAS BEEN TAKEN FROM THE LAB FUNCTIONS
# PROVIDED IN THE COURSE LAB
# 
# NPREPROCESSING_removePunctuation()
#
# INPUT: String - fieldName - name of field
#
# OUTPUT : String - name of field with punctuation removed
# ************************************************
NPREPROCESSING_removePunctuation<-function(fieldName){
  return(gsub("[[:punct:][:blank:]]+", "", fieldName))
}

# ************************************************
# THIS FUNCTION HAS BEEN TAKEN FROM THE LAB FUNCTIONS
# PROVIDED IN THE COURSE LAB
# 
# NreadDataset() :
#
# Read a CSV file from working directory
#
# INPUT: string - csvFilename - CSV filename
#
# OUTPUT : data frame - contents of the headed CSV file
# ************************************************
NreadDataset<-function(csvFilename){
  
  dataset<-read.csv(csvFilename,encoding="UTF-8",stringsAsFactors = FALSE)
  
  # The field names "confuse" some of the library algorithms
  # As they do not like spaces, punctuation, etc.
  names(dataset)<-NPREPROCESSING_removePunctuation(names(dataset))
  
  print(paste("CSV dataset",csvFilename,"has been read. Records=",nrow(dataset)))
  return(dataset)
}

# ************************************************
# ConfusionMat(model, training_data, testing_data) :
#
# Provide a confusion matrix for testing and training data
#
# INPUT: model - ML model
#        training_data - the training dataset
#        testing_data - the testing dataset
#
# OUTPUT : confusion matrix list
# ************************************************
ConfusionMat <- function(model, training_data, testing_data) {
  
  # confusion matrix for training data
  train_predictions <- predict(model, training_data)
  cm_train <- confusionMatrix(train_predictions, training_data$Attrition)
  print(cm_train)
  
  # confusion matrix for testing data
  test_prediction <- predict(model, testing_data)
  cm_test <- confusionMatrix(test_prediction, testing_data$Attrition)
  print(cm_test)
  
  confusion_matrices <- list()
  confusion_matrices <- c(list(cm_train), list(cm_test))
  
  return(confusion_matrices)
}

# ************************************************
# ClassificationReport(conf_matrices_list) :
#
# Provide a classification report for testing data
#
# INPUT: list - confusion matrices list
#
# OUTPUT : classification report for testing data
# ************************************************

ClassificationReport <- function(conf_matrices_list) {
  
  confusion_matrix_test <- conf_matrices_list[2][[1]]
  
  # Calculate precision, accuracy, recall, and F1 score
  precision <- confusion_matrix_test$byClass["Pos Pred Value"]
  accuracy <- confusion_matrix_test$overall["Accuracy"]
  recall <- confusion_matrix_test$byClass["Sensitivity"]
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  classification_report_testing <- data.frame(
    Accuracy = confusion_matrix_test$overall['Accuracy'],
    Recall = recall,
    Precision = precision,
    F1_Score = f1_score
  )
  return(classification_report_testing)
}

# ************************************************
# ROC_AUC_Plot(testing_data, test_predictions, model_name) :
#
# Plots ROC curve for models and provides AUC
#
# INPUT: testing_data - a testing dataset for calculating ROC-AUC
#        test_predictions - model predictions for calculating ROC-AUC
#        model_name - name of the model for calculating ROC-AUC
#
# OUTPUT : plots ROC curve
# ************************************************
ROC_AUC_Plot <- function(testing_data, test_predictions, model_name) {
  roc_obj <- roc(testing_data$Attrition, test_predictions)
  auc_value <- auc(roc_obj)
  
  # Plot the ROC curve
  plot(roc_obj, main = paste("ROC Curve for", model_name, "Model"))
  text(0.2, 0.2, paste("AUC = ", round(auc_value, 4)), cex = 1.4)
}

# ************************************************
# prepare_classification_report(conf_mat, model_name) :
#
# Creates a classification report from the passed confusion matrix
#
# INPUT: conf_mat - a confusion matrix
#.       model_name - model for which classification report is created
#
# OUTPUT : classification report
# ************************************************
prepare_classification_report <- function(confusion_matrix_test, model_name) {
  
  precision <- confusion_matrix_test$byClass["Pos Pred Value"]
  accuracy <- confusion_matrix_test$overall["Accuracy"]
  recall <- confusion_matrix_test$byClass["Sensitivity"]
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  classification_report_testing <- data.frame(
    Accuracy = accuracy,
    Recall = recall,
    Precision = precision,
    F1_Score = f1_score
  )
  borderPrint(paste("Print Classification Report for", model_name))
  print(classification_report_testing)
  
  return(classification_report_testing)
}

# ************************************************
# implement_SVM (training_data, testing_data) :
#
# Implements the Support Vector Machine
#
# INPUT: training_data - data frame - dataset for training the model
#        testing_data - data frame - dataset for testing the model
#
# OUTPUT : Support Vector Machine model's classification report
# ************************************************
implement_SVM <- function(training_data, testing_data) {
  
  svm_model <- svm(Attrition ~ ., data = training_data, kernel = "radial", cost = 10, gamma = 0.01)
  
  train_predictions_svm = predict(svm_model, newdata = training_data, type = "class")
  
  # Generate the confusion matrix
  confusion_matrix_train_svm <- confusionMatrix(train_predictions_svm, training_data$Attrition)
  borderPrint("Printing confusion Matrix for SVM training data")
  print(confusion_matrix_train_svm)
  
  test_predictions_svm <- predict(svm_model, testing_data, type = "class")
  
  # confusion matrix for testing data
  confusion_matrix_test_svm <- confusionMatrix(test_predictions_svm, testing_data$Attrition)
  borderPrint("Printing confusion Matrix for SVM testing data")
  print(confusion_matrix_test_svm)
  
  # Creating classification report
  classification_report_testing_svm <- prepare_classification_report(confusion_matrix_test_svm,
                                                                     "SVM")
  return(classification_report_testing_svm)
} 

# ************************************************
# implement_KNN(training_data, testing_data) :
#
# Implements the K-Nearest Neighbour Model
#
# INPUT: training_data - data frame - dataset for training the model
#        testing_data - data frame - dataset for testing the model
#
# OUTPUT : K-Nearest Neighbor Model's classification report
# ************************************************
implement_KNN <- function(training_data, testing_data) {
  
  k_value <- round(sqrt(nrow(training_data)))   #common starting point of setting k
  prediction <- knn(train=training_data[, -which(names(training_data) == "Attrition")],
                    test = testing_data[, -which(names(testing_data) == "Attrition")],
                    cl = training_data[, which(names(training_data) == "Attrition")],
                    k = k_value)
  
  borderPrint("Creating Confusion Matrices for kNN Model")
  conf_matrix_kNN <- confusionMatrix(prediction, testing_data[, which(names(training_data) == "Attrition")])
  
  # Creating classification report
  classification_report_testing_knn <- prepare_classification_report(conf_matrix_kNN,
                                                                    "kNN")
  return(classification_report_testing_knn)
}

# ************************************************
# implement_Naive_Bayes (training_data, testing_data) :
#
# Implements the Naive Bayes Model
#
# INPUT: training_data - data frame - dataset for training the model
#.       testing_data - data frame - dataset for testing the model
#
# OUTPUT : Naive Bayes ML Model's classification report
# ************************************************
implement_Naive_Bayes <- function(training_data, testing_data) {
  
  ctrl <- trainControl(method = "cv", number = 7)
  naive_bayes_model <- train(Attrition ~ ., data = training_data,
                             method = "naive_bayes", trControl = ctrl,
                             preProc = c("BoxCox", "center", "scale", "pca"))
  
  test_predictions_NB <- predict(naive_bayes_model, testing_data)
  
  # ************     Creating ROC for Naive Bayes     ************
  
  NB_test_predictions_ROC <- predict(naive_bayes_model, testing_data, type = "prob")[, 2]
  ROC_AUC_Plot(testing_data, NB_test_predictions_ROC, "Naive Bayes")
  
  # ************ Closing section: ROC for Naive Bayes ************
  
  # confusion matrix for testing data
  confusion_matrix_test_NB <- confusionMatrix(test_predictions_NB, testing_data$Attrition)
  print("Printing confusion Matrix for Naive Bayes testing data")
  print(confusion_matrix_test_NB)
  
  # Creating classification report
  classification_report_testing_NB <- prepare_classification_report(confusion_matrix_test_NB,
                                                                    "Naive Bayes")
  
  return(classification_report_testing_NB)
}

# ************************************************
# implement_neural_network (dataset, numeric_cols) :
#
# Implements the Neural Network
#
# INPUT: dataset - a dataset for which ML model is implemented
#        numeric_cols - a list of numeric cols in the dataset
#
# OUTPUT : Classification Report and performance score of Neural Network model
# ************************************************
implement_neural_network <- function(dataset, numeric_cols) {
  
  borderPrint("Scaling the numerical features through z-scale normalization.")
  numeric_cols <- c(colnames(numeric_cols), "OverTime")
  # Find the common column names that exist in the data frame
  existing_cols <- intersect(numeric_cols, colnames(dataset))
  
  # Check if there are any missing columns
  missing_cols <- setdiff(numeric_cols, colnames(dataset))
  
  if (length(missing_cols) > 0) {
    cat("The following columns are not present in the data frame:", paste(missing_cols, collapse = ", "), "\n")
  }
  
  # Perform z-score normalization on the specified columns that exist
  if (length(existing_cols) > 0) {
    dataset[, existing_cols] <- scale(dataset[, existing_cols])
  }
  
  borderPrint("Setting the data types for Neural Network model")
  X <- as.matrix(dataset[, -which(names(dataset) == "Attrition")])
  y <- as.numeric(dataset$Attrition)
  
  borderPrint("Splitting the data into training and testing sets.")
  set.seed(123)  # for reproducibility
  indexes <- createDataPartition(y, p = 0.8, list = FALSE)
  X_train <- X[indexes, ]
  y_train <- y[indexes]
  X_test <- X[-indexes, ]
  y_test <- y[-indexes]
  
  borderPrint("Defining the Neural Network Model")
  Neural_Network_model <- keras_model_sequential() %>%
    layer_dense(units = ncol(X_train), activation = 'relu', input_shape = c(ncol(X_train))) %>%
    layer_dropout(0.2) %>%
    layer_dense(units = ncol(X_train), activation = 'relu') %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 1, activation = 'sigmoid')  # For binary classification
  
  Neural_Network_model %>% compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
  
  borderPrint("Training the Neural Network Model")
  NN_training <- Neural_Network_model %>% fit(
    X_train,
    y_train,
    epochs =60,
    batch_size = 32,
    validation_split = 0.2
  )
  
  borderPrint("Evaluating the Neural Network Model")
  # Evaluating the model
  performance_score <- Neural_Network_model %>% evaluate(X_test, y_test)
  
  # Making predictions
  borderPrint("Making predictions")
  predictions <- predict(Neural_Network_model, X_test)
  predicted_classes <- ifelse(predictions > 0.5, 1, 0)
  
  borderPrint("Creating the Confusion Matrix for testing data")
  # confusion matrix for testing data
  confusion_matrix_test_nn <- confusionMatrix(as.factor(predicted_classes),
                                              as.factor(y_test))
  borderPrint("Printing confusion Matrix for Neural Networks testing data")
  print(confusion_matrix_test_nn)
  
  # Creating classification report
  classification_report_testing_nn <- prepare_classification_report(confusion_matrix_test_nn,
                                                                    "Neural Network")
  NN_metrics <- c(classification_report_testing_nn, performance_score)
  return(NN_metrics)
}

# ************************************************
# implement_neural_network_cv (dataset, numeric_cols) :
#
# Implements the Neural Network with cross-validation
#
# INPUT: dataset - a dataset for which ML model is implemented
#        numeric_cols - a list of numeric cols in the dataset
#.       cv - bool value to perform cross validation
#
# OUTPUT : Classification Report and avg performance of Neural Network model
# ************************************************
implement_neural_network_cv <- function(dataset, numeric_cols){
  
  borderPrint("Scaling the numerical features through z-scale normalization.")
  numeric_cols <- c(colnames(numeric_cols), "OverTime")
  # Find the common column names that exist in the data frame
  existing_cols <- intersect(numeric_cols, colnames(dataset))
  
  # Check if there are any missing columns
  missing_cols <- setdiff(numeric_cols, colnames(dataset))
  
  if (length(missing_cols) > 0) {
    cat("The following columns are not present in the data frame:", paste(missing_cols, collapse = ", "), "\n")
  }
  
  # Perform z-score normalization on the specified columns that exist
  if (length(existing_cols) > 0) {
    dataset[, existing_cols] <- scale(dataset[, existing_cols])
  }
  
  X <- as.matrix(dataset[, -which(names(dataset) == "Attrition")])
  y <- as.numeric(dataset$Attrition)
  # Define the number of folds if cv is set to TRUE
  k<-5
  set.seed(123)  # for reproducibility
  folds <- createFolds(y, k = k, list = TRUE)
  
  # Store results
  cv_results <- list()
  
  # Loop over each fold
  for(i in seq_along(folds)) {
    borderPrint(paste("Training on fold", i))
    
    # Split data into training and validation set
    training_indices <- folds[[i]]
    validation_indices <- setdiff(1:length(y), training_indices)
    
    X_train_cv <- X[training_indices, ]
    y_train_cv <- y[training_indices]
    X_val_cv <- X[validation_indices, ]
    y_val_cv <- y[validation_indices]
    
    # Define the Neural Network Model
    Neural_Network_model <- keras_model_sequential() %>%
      layer_dense(units = ncol(X_train_cv), activation = 'relu', input_shape = c(ncol(X_train_cv))) %>%
      layer_dropout(0.2) %>%
      layer_dense(units = ncol(X_train_cv), activation = 'relu') %>%
      layer_dropout(0.2) %>%
      layer_dense(units = 1, activation = 'sigmoid')
    
    Neural_Network_model %>% compile(
      optimizer = 'adam',
      loss = 'binary_crossentropy',
      metrics = c('accuracy')
    )
    
    # Train the Neural Network Model
    NN_training <- Neural_Network_model %>% fit(
      X_train_cv,
      y_train_cv,
      epochs = 60,
      batch_size = 32,
      validation_data = list(X_val_cv, y_val_cv)
    )
    
    # Evaluate the model on the validation set
    score <- Neural_Network_model %>% evaluate(X_val_cv, y_val_cv)
    cv_results[[i]] <- score
  }
  
  # Calculate average performance metrics across all folds
  cv_performance <- do.call(rbind, cv_results)
  average_performance <- colMeans(cv_performance, na.rm = TRUE)
  
  # Output the average performance
  print(average_performance)
  
  # Making predictions
  borderPrint("Making predictions")
  predictions <- predict(Neural_Network_model, X_val_cv)
  predicted_classes <- ifelse(predictions > 0.5, 1, 0)
  
  borderPrint("Creating the Confusion Matrix for testing data")
  confusion_matrix_test_nn <- confusionMatrix(as.factor(predicted_classes),
                                              as.factor(y_val_cv))
  borderPrint("Printing confusion Matrix for Neural Networks testing data")
  print(confusion_matrix_test_nn)
  
  # Creating classification report
  classification_report_testing_nn <- prepare_classification_report(confusion_matrix_test_nn,
                                                                    "Neural Network")
  NN_metrics <- c(classification_report_testing_nn, average_performance)
  return(NN_metrics)
  
}

# ************************************************
# implement_decision_tree (training_data, testing_data) :
#
# Implements the Decision Tree model
#
# INPUT: training_data - dataset used for training the model
#        testing_data - dataset used for testing the model
#
# OUTPUT : Classification Report of Decision Tree model
# ************************************************
implement_decision_tree <- function(training_data, testing_data) {
  
  dt_model= rpart(Attrition~., data = training_data, method="class", minbucket = 25)
  
  # ************* Analyzing the Decision Tree Feature Importance *************
  
  plot(dt_model, uniform=TRUE, branch=0.6, margin=0.05)
  text(dt_model, all=TRUE, use.n=TRUE)
  title("Training Set's Classification Tree")
  
  dt_importance <- data.frame(dt_model$variable.importance)
  colnames(dt_importance)[colnames(dt_importance) == "dt_model.variable.importance"] <- "weightage"
  dt_importance$predictors <- rownames(dt_importance)
  total_importance <- sum(dt_importance$weightage)
  dt_importance$percentage <- dt_importance$weightage / total_importance * 100
  
  ggplot(dt_importance, aes(x = dt_importance$weightage, y = reorder(dt_importance$predictors, dt_importance$weightage))) +
    geom_bar(stat = "identity", fill = "red") +
    geom_text(aes(label = paste0(round(dt_importance$percentage, 2), "%")), hjust = -0.1, size = 4) +
    labs(title = "Feature Importance", x = "Weightage", y = "Feature") +
    scale_x_continuous(labels = scales::percent_format(scale = 1)) +
    theme_minimal()
  
  # ************* Close of the Decision Tree Feature Importance *************
  
  # Generate the confusion matrix for testing data
  test_predictions_dt <- predict(dt_model, testing_data, type = "class")
  
  confusion_matrix_test_dt <- confusionMatrix(test_predictions_dt, testing_data$Attrition)
  print("Printing confusion Matrix for DT testing data")
  print(confusion_matrix_test_dt)
  
  # **************** Creating ROC for Decision Tree ****************
  
  DT_test_predictions_ROC <- predict(dt_model, testing_data, type = "prob")[, 2]
  ROC_AUC_Plot(testing_data, DT_test_predictions_ROC, "Decision Tree")
  
  # ************ Closing section: ROC for Decision Tree ************
  
  # Calculate precision, accuracy, recall, and F1 score for DT Testing Data Results
  classification_report_testing_dt <- prepare_classification_report(confusion_matrix_test_dt,
                                                                         "Random Forest")
  
  return(classification_report_testing_dt)
}

# ************************************************
# implement_random_forest (training_data, testing_data) :
#
# Implements the Random Forest model
#
# INPUT: training_data - dataset used for training the model
#        testing_data - dataset used for testing the model
#
# OUTPUT : Classification Report of Random Forest model
# ************************************************
implement_random_forest <- function(training_data, testing_data) {
  
  # Rename the variable to remove hyphen and make the ready ready for randomForest model processing
  names(training_data)[names(training_data) == 'BusinessTravelNon-Travel'] <- 'BusinessTravelNon_Travel'
  training_data_RF <- training_data
  names(training_data_RF) <- gsub("[- ]", "_", names(training_data))
  names(training_data_RF) <- gsub("&", "and", names(training_data_RF))
  
  # Creating a Random Forest model
  randomForestModel <- randomForest(Attrition ~ ., data = training_data_RF, ntree = 100, nodesize = 10)
  
  # Processing the testing data variable names
  testing_data_RF <- testing_data
  names(testing_data_RF) <- gsub("[- ]", "_", names(testing_data_RF))
  names(testing_data_RF) <- gsub("&", "and", names(testing_data_RF))
  
  RF_test_predictions <- predict(randomForestModel, newdata=testing_data_RF)
  # confusion matrix for testing data
  RF_confusion_matrix_test <- confusionMatrix(RF_test_predictions, testing_data_RF$Attrition)
  print("Printing confusion Matrix for Random Forest testing data")
  print(RF_confusion_matrix_test)
  
  
  # ************     Creating ROC for Random Forest     ************
  RF_test_predictions_ROC <- predict(randomForestModel, testing_data_RF, type = "prob")[, 2]
  ROC_AUC_Plot(testing_data_RF, RF_test_predictions_ROC, "Random Forest")
  
  # ************ Closing section: ROC for Random Forest ************
  
  # Creating classification report
  RF_classification_report_testing <- prepare_classification_report(RF_confusion_matrix_test,
                                                                     "Random Forest")
  
  return(RF_classification_report_testing)
}

# ************************************************
# plot_avg_against_feature (dataset, feature_x, feature_y) :
#
# Plots average of feature_x against feature_y
#
# INPUT: dataset - data set 
#        feature_x - feature to plot the average of
#        feature_y - feature against which the feature_x is plotted
#
# OUTPUT : Bar plot
# ************************************************

plot_avg_against_feature <- function(dataset, feature_x, feature_y) {
  
  # Calculating the average of feature_x
  average_feature_x <- dataset %>% group_by(.data[[feature_y]]) %>%
                       summarize(avg_feature_x = mean(.data[[feature_x]]))
  
  # Plotting average feature_x for feature_y
  gg_plot <- ggplot(average_feature_x, aes_string(x = feature_y, y = "avg_feature_x", fill = feature_y)) +
    geom_col() +
    geom_text(aes(label = round(avg_feature_x, 2)),
              vjust = -0.8,
              size = 5.5) + 
    labs(title = paste("Average", feature_x, "by", feature_y),
         x = feature_y,
         y = paste("Average ", feature_x)) +
    theme_minimal() +
    scale_fill_brewer(palette = "Spectral")
  
  return(gg_plot)
}

# ************************************************
# main() :
# main entry point to execute analytics
#
# INPUT       :   None
#
# OUTPUT      :   None
#
# Keeps all objects as local to this function
# ************************************************
main<-function(){
  
  borderPrint("Starting the Practical Business Analytics Project: Employee Attrition Analysis")
  
  print(DATASET_FILENAME)
  
  employees<-NreadDataset(DATASET_FILENAME)
  # view the head and tail
  print(head(employees))
  print(tail(employees) )
  
  # view a random sample
  print(sample_n(employees,size=10))

  # Find the current number of records and features
  num_rows <- nrow(employees)
  print(paste("Total No. Of Records=", num_rows))
  num_columns <- ncol(employees)
  print(paste("Total No. of Features=", num_columns))
  
  # Find the  Count of Unique Values for Character Variables
  unique_counts <- lapply(employees, function(x) length(unique(x)))
  print(unique_counts)
  
  # Identify and separate the numerical and non-numerical features
  numeric_cols <- employees[sapply(employees, is.numeric)]
  non_numeric_cols <- employees[!sapply(employees, is.numeric)]
  # view the summary of numerical features
  print(summary(numeric_cols))
  
  # Check for missing values
  missingValues <- checkMissingValues(employees)
  # Print the missing data (if any)
  if (!is.null(missingValues)) {
    borderPrint(missingValues)
  } else {
    borderPrint("No missing values found in the dataset")
  }
  
  # Check for the duplicates in rows and columns
  duplicates <- checkForDuplicates(employees)
  if (any(duplicates)) {
    borderPrint("Duplicate rows found:")
    print(employees[duplicate_rows, ])
  } else {
    borderPrint("No duplicate rows found.")
  }
  
  # Finding out unique values in each feature and plot their graphs
  par(mfrow=c(1, 1))
  for (col in colnames(numeric_cols)){
    hist(numeric_cols[[col]], main = col, xlab = col, ylab = "Frequency", col = "lightblue", border = "black")
  }
  
  # After plotting, dropping a few features that do not add value i.e., reducing dimensionality
  # We are dropping four features: 
  # Over18:               it contains a single value - "Yes". Does not provide any value addition. 
  # Employee count:       it contains a single value for all records
  # Employee number:      is unique but adds no value towards predicting attrition
  # StandardHours:        it contains a single value for all records
  
  # Get the column index for the features to drop
  features_to_drop <- c("EmployeeCount", "EmployeeNumber", "Over18", "StandardHours")
  employees_trimmed <- employees[, !names(employees) %in% features_to_drop]
  
  # storing dataset for CatBoost Algorithm
  employees_data_wo_label_encoding <- employees_trimmed

  
  # ****************************************************************
  # ****************************************************************
                              # EDA Section
  # ****************************************************************
  # ****************************************************************
 
  # Identify the unique values and their counts for Attrition Feature
  unique_counts <- table(employees_trimmed['Attrition'])

  yes_count <- sum(employees_trimmed$Attrition == "Yes")
  no_count <- sum(employees_trimmed$Attrition == "No")
  print(yes_count)
  print(no_count)
  
  labels <- data.frame(
    category = factor(c("Yes", "No")),
    count = c(yes_count, no_count)
  )
  
  # ****************************************************************
  # Converting numerical to categorical text values for EDA
  
  employees_num_2_categorical <- employees_trimmed
  
  employees_num_2_categorical$Education <- factor(employees_num_2_categorical$Education, 
                                                  levels = 1:5, 
                                                  labels = c('Below College', 'College', 'Bachelor', 'Master', 'Doctor'))
  
  employees_num_2_categorical$EnvironmentSatisfaction <- factor(employees_num_2_categorical$EnvironmentSatisfaction, 
                                                                levels = 1:4, 
                                                                labels = c('Low', 'Medium', 'High', 'Very High'))
  
  employees_num_2_categorical$JobInvolvement <- factor(employees_num_2_categorical$JobInvolvement, 
                                                       levels = 1:4, 
                                                       labels = c('Low', 'Medium', 'High', 'Very High'))
  
  employees_num_2_categorical$JobSatisfaction <- factor(employees_num_2_categorical$JobSatisfaction, 
                                                        levels = 1:4, 
                                                        labels = c('Low', 'Medium', 'High', 'Very High'))
  
  employees_num_2_categorical$PerformanceRating <- factor(employees_num_2_categorical$PerformanceRating, 
                                                          levels = 1:4, 
                                                          labels = c('Low', 'Good', 'Excellent', 'Outstanding'))
  
  employees_num_2_categorical$RelationshipSatisfaction <- factor(employees_num_2_categorical$RelationshipSatisfaction, 
                                                                 levels = 1:4, 
                                                                 labels = c('Low', 'Medium', 'High', 'Very High'))
  
  employees_num_2_categorical$WorkLifeBalance <- factor(employees_num_2_categorical$WorkLifeBalance, 
                                                        levels = 1:4, 
                                                        labels = c('Bad', 'Good', 'Better', 'Best'))
  
  
  # ****************************************************************
  #         Univariate Analysis: Plotting features individually 
  # ****************************************************************
  
  features_to_plot <- c("Attrition", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")
  for (feature in features_to_plot){
    borderPrint(paste("Plotting Feature:", feature))
    createChart(employees_trimmed, feature)
  }

  
  # ****************************************************************
  #         Bivariate analysis - plotting two features together
  # ****************************************************************
  
  # bar chart with two features
  featuresToPlotWithAttrition <- c("BusinessTravel", "Department", "EducationField", "Gender",
                                   "JobRole", "MaritalStatus", "OverTime", "EnvironmentSatisfaction",
                                   "JobInvolvement", "JobSatisfaction", "RelationshipSatisfaction",
                                   "PerformanceRating", "WorkLifeBalance", "StockOptionLevel",
                                   "Age", "DistanceFromHome", "MonthlyIncome", "NumCompaniesWorked",
                                   "PercentSalaryHike", "TotalWorkingYears", "TrainingTimesLastYear",
                                   "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion")
  
  plots <- plotTwoFeatures(employees_trimmed, featuresToPlotWithAttrition, "Attrition")
  for (i in 1:length(plots)) {
    # Display each bar chart one by one
    print(plots[[i]])
  }
  
  plot_obj <- plot_two_features_against_attrition_stacked(employees_trimmed, "Gender", "Department")
  ggsave("stacked_bar_plot.png", plot = plot_obj, width = 4, height = 14, dpi = 300)
  print(plot_obj)
  
  # Multi-variate Analysis
  ggplot(employees_trimmed, aes(x = Gender, y = Age, fill = Attrition)) +
    geom_boxplot() +
    scale_fill_manual(values = c("Yes" = "red", "No" = "green")) +
    labs(title = "Age Distribution by Gender and Attrition",
         x = "Gender",
         y = "Age") +
    theme_minimal()
  
  ggplot(employees_trimmed, aes(x = Age, fill = Attrition)) +
    geom_bar(position = "dodge") +  # 'dodge' position to place bars side by side
    geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5, position = position_dodge(width = 0.9), size = 3.5) +
    scale_fill_manual(values = c("Yes" = "red", "No" = "green")) +
    labs(title = "Attrition Count by Age",
         x = "Age",
         y = "Count") +
    
    theme_minimal()
  
  # Create the density plot for Age Distribution for Male and Female
  data <- employees_trimmed
  mean_age_female <- round(mean(data$Age[data$Gender == 'Female']), 2)
  mean_age_male <- round(mean(data$Age[data$Gender == 'Male']), 2)
  mean_age_overall <- mean(data$Age)
  
  mean_ages <- data.frame(
    Gender = c("Female", "Male"),
    MeanAge = c(mean_age_female, mean_age_male),
    Label = c(paste("Mean =", mean_age_female, "Years Old"),
              paste("Mean =", mean_age_male, "Years Old"))
  )
  
  ggplot(data, aes(x = Age, fill = Gender)) + 
    geom_density(alpha = 0.6) +  # Adjust transparency with alpha
    facet_wrap(~Gender, scales = 'free') +  # Create separate plots for each gender
    geom_vline(data = mean_ages[mean_ages$Gender == "Female",], aes(xintercept = mean_age_female), color = "red") + 
    geom_vline(data = mean_ages[mean_ages$Gender == "Male",], aes(xintercept = mean_age_male), color = "red") + 
    geom_text(data = mean_ages, aes(x = MeanAge, label = Label, y = 0.02),
              vjust = -15, hjust = -0.5, size = 5) +
    # annotate("text", label = paste("Mean =", mean_age_female, "Years Old"), x = 50, y = 0.03, color = "black")
    theme_minimal() + 
    labs(title = "Age Distribution",
         x = "Age",
         y = "Density") + 
    scale_fill_manual(values = c("Female" = "pink", "Male" = "blue"))
  
  # ********************************************************
  # Create a plot for Average Monthly Income for each department
  # ********************************************************
  income_dpt_plot <- plot_avg_against_feature(employees_trimmed, "MonthlyIncome", "Department")
  print(income_dpt_plot)
  
  # ********************************************************
  # Create a plot for Average Job Satisfaction in each department
  # ********************************************************
  job_satisfaction_dpt_plot <- plot_avg_against_feature(employees_trimmed, "JobSatisfaction", "Department")
  plot(job_satisfaction_dpt_plot)
  
  # ********************************************************
  # Create a plot for Average Job Satisfaction in each Job Role
  # ********************************************************
  job_satisfaction_job_role_plot <- plot_avg_against_feature(employees_trimmed, "JobSatisfaction", "JobRole")
  plot(job_satisfaction_job_role_plot)

  # ********************************************************
  # Create a plot for Average Monthly Income in each Job Role
  # ********************************************************
  monthlyIncome_JobRole_plot <- plot_avg_against_feature(employees_trimmed, "MonthlyIncome", "JobRole")
  plot(monthlyIncome_JobRole_plot)
  
  # ********************************************************
  # Create a plot for Average Environment Satisfaction in each Job Role
  # ********************************************************
  envSatisfaction_JobRole_plot <- plot_avg_against_feature(employees_trimmed, "EnvironmentSatisfaction", "JobRole")
  plot(envSatisfaction_JobRole_plot)
  
  # ********************************************************
  # Create a plot for Average WorkLifeBalance in each Job Role
  # ********************************************************
  workLifeBalance_JobRole_plot <- plot_avg_against_feature(employees_trimmed, "WorkLifeBalance", "JobRole")
  plot(workLifeBalance_JobRole_plot)
  
  # ********************************************************
  # Create a plot for Current Manager and Job Satisfaction
  # ********************************************************
  
  # stacked bar chart
  ggplot(employees_num_2_categorical, aes(x = as.factor(YearsWithCurrManager), fill = as.factor(JobSatisfaction))) +
    geom_bar() +
    labs(title = "Representation of Job Satisfaction by Years With Current Manager",
         x = "Years With Current Manager",
         y = "Count",
         fill = "Job Satisfaction") +
    theme_minimal() +
    scale_fill_brewer(palette = "RdYlBu")
  
  # faceted bar chart
  ggplot(employees_num_2_categorical, aes(x = as.factor(EnvironmentSatisfaction), y =)) +
    geom_bar() +
    facet_wrap(~JobRole) +
    scale_fill_brewer(palette = "RdYlBu") +
    labs(title = "Distribution of Environmental Satisfaction for Each Job Role",
         x = "Environmental Satisfaction",
         y = "Count") +
    theme_minimal()
  
    
  # *********************************************************************
  # *********************************************************************
  
  #                         Feature Engineering
  
  # *********************************************************************
  # *********************************************************************
  
  # mapping Yes and No to 1 and 0 respectively - label encoding 
  employees_trimmed$Attrition <- ifelse(employees_trimmed$Attrition == "Yes", 1, 0)
  employees_trimmed$OverTime <- ifelse(employees_trimmed$OverTime == "Yes", 1, 0)
  
  # label encode the age by first defining the age groups
  employees_trimmed$Age <- ifelse(employees_trimmed$Age >= 18 & employees_trimmed$Age < 30, 
                                  "Juniors",
                                  ifelse(employees_trimmed$Age >= 30 & employees_trimmed$Age < 45,
                                         "Mid_Level_Seniors",
                                         ifelse(employees_trimmed$Age >= 45 & employees_trimmed$Age <= 60,
                                                "Seniors",
                                                "None")))
  
  employees_data_with_label_encoding <- employees_trimmed
  
  trimmed_numeric_cols <- employees_trimmed[sapply(employees_trimmed, is.numeric)]
  trimmed_non_numeric_cols <- employees_trimmed[!sapply(employees_trimmed, is.numeric)]
  
  # dropping attrition from numeric cols
  NC_withoutAttrition <- trimmed_numeric_cols[-which(names(trimmed_numeric_cols) == "Attrition")]
  
  # Perform 1-hot-encoding on categorical features
  # Creating dummy variables transformations
  dummy <- dummyVars("~ .", data = trimmed_non_numeric_cols)
  
  # Applying the transformation
  one_hot_encoded_data <- predict(dummy, newdata = trimmed_non_numeric_cols)
  
  # Convert to a dataframe
  one_hot_encoded_cat_data <- as.data.frame(one_hot_encoded_data)
  
  # combining the numerical and one hot encoded data together
  combined_data <- cbind(NC_withoutAttrition, one_hot_encoded_cat_data)   # Data without Attrition
  
  # ****************************************************************** 
  # ******************************************************************
  # Balancing the dataset via Synthetic Minority Over-sampling Technique (SMOTE)
  
  attrition_encoded <- trimmed_numeric_cols[which(names(trimmed_numeric_cols) == "Attrition")]
  combined_data_with_attrition <- cbind(combined_data, attrition_encoded)
  
  oversampled_data <- smote_balancing(combined_data_with_attrition, "Attrition")
  oversampled_data_wo_attrition <- oversampled_data[, -which(names(oversampled_data) == "Attrition")]
  
  oversampled_yes_count <- sum(oversampled_data$Attrition == "1")
  oversampled_no_count <- sum(oversampled_data$Attrition == "0")
  print(oversampled_yes_count)
  print(oversampled_no_count)
  
  # creating a heatmap
  correlation_matrix <- cor(trimmed_numeric_cols)
  abs_correlation_matrix <- abs(correlation_matrix)
  
  long_cor <- reshape2::melt(abs_correlation_matrix)
  
  heatmap_ <-ggplot(long_cor, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.2f", value)), size = 3, color = "black") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                         midpoint = 0, limit = c(-1,1)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(fill = "Correlation")
  
  print(heatmap_)
  
  less_correlated_features <- c("NumCompaniesWorked", "MonthlyRate", 
                                "HourlyRate", "Education",
                                "DistanceFromHome", "DailyRate", "RelationshipSatisfaction",
                                "TrainingTimeLastYear", "WorkLifeBalance", "Attrition")
  
  correlated_numeric_col <- trimmed_numeric_cols[, !names(trimmed_numeric_cols) %in% less_correlated_features]
  
  combined_data_wo_attrition <- cbind(correlated_numeric_col, one_hot_encoded_cat_data)  # Dataset without Attrition
  
  # scaling the dataframe
  # removing attrition as there is no need to scale the attrition feature
  x <- MinMaxScaling(oversampled_data_wo_attrition)
  y <- oversampled_data[which(names(oversampled_data) == "Attrition")]    # Attrition column in y
  
  CombinedDateForML <- cbind(x, y)      # Combining Scaled features with Attrition
  
  # randomize and split the dataset
  set.seed(231)
  
  CombinedDateForML$Attrition <- as.factor(CombinedDateForML$Attrition)
  train_index <- createDataPartition(y = CombinedDateForML$Attrition,  # y = our dependent variable.
                                     p = .7,  # Specifies split into 70% & 30%.
                                     list = FALSE,  # Sets results to matrix form. 
                                     times = 1)  # Sets number of partitions to create to 1. 
  training_data <- CombinedDateForML[train_index,]  # Use train_index of Employee Attrition data to create train_data.
  testing_data <- CombinedDateForML[-train_index,]  # Use whatever that is not in train_index to create test_data.
  
  # *********************************************************************
  # *********************************************************************
  #                         Machine Learning Models
  # *********************************************************************
  # *********************************************************************
  
  
  # *********************************************************************
  #                     Logistic Regression Model
  # *********************************************************************
  
  fitControl <- trainControl(method = "cv", number = 10, savePredictions = TRUE)
  # Define the model
  lr_model <- train(Attrition ~ .,
                    data = training_data,
                    method = 'glm',
                    family = 'binomial',
                    trControl = fitControl)
  
  test_predictions_lr <- predict(lr_model, testing_data)
  
  # confusion matrix for testing data
  confusion_matrix_test_lr <- confusionMatrix(test_predictions_lr, testing_data$Attrition)
  borderPrint("Printing confusion Matrix for SVM testing data")
  print(confusion_matrix_test_lr)
  
  # Creating classification report
  LR_classification_report_testing <- prepare_classification_report(confusion_matrix_test_lr,
                                                                     "Logistic Regression")
  
  # *********************************************************************
  #                           Decision Tree Model
  # *********************************************************************
  
  DT_classification_report_testing <- implement_decision_tree(training_data, testing_data)  
  
  # *********************************************************************  
  #                           Support Vector Machine Model
  # *********************************************************************
  
  SVM_classification_report_testing <- implement_SVM(training_data, testing_data)
  
  # ********************************************************************* 
  #                           Random Forest Model
  # *********************************************************************
  
  RF_classification_report_testing <- implement_random_forest(training_data, testing_data)
  
  # *********************************************************************  
  #                            Neural Networks Model
  # *********************************************************************
  # Running the model without cross-validation
  NN_metrics <- implement_neural_network(oversampled_data, numeric_cols)
  NN_classification_report_wo_cv <- NN_metrics[1:4]
  performance_score <- NN_metrics[5:6]
  # Running the model with cross-validation to enhance performance
  NN_metrics_with_cv <- implement_neural_network_cv(oversampled_data, numeric_cols)
  NN_classification_report_with_cv <- NN_metrics_with_cv[1:4]
  performance_score_with_cv <- NN_metrics_with_cv[5:6]
  

  # *********************************************************************  
  #                           K-Nearest Neighbour Model
  # *********************************************************************
  KNN_classification_report_testing <- implement_KNN(training_data, testing_data)
  
  # *********************************************************************  
  #                           Naive Bayes Model
  # *********************************************************************
  NB_classification_report_testing <- implement_Naive_Bayes(training_data, testing_data)
  
  # *********************************************************************
  #           Combining all the classficiation Reports together
  # *********************************************************************
  classification_reports_combined <- list()
  classification_reports_combined <- c(list(LR_classification_report_testing),
                                       list(DT_classification_report_testing),
                                       list(RF_classification_report_testing),
                                       list(SVM_classification_report_testing),
                                       list(KNN_classification_report_testing),
                                       list(NB_classification_report_testing),
                                       list(NN_classification_report_testing))
  
  summary_df <- data.frame(
    Model = character(),
    Accuracy = numeric(),
    Precision = numeric(),
    Recall = numeric(),
    F1_Score = numeric(),
    stringsAsFactors = FALSE
  )
  
  model_names <- c("Logistic Regression", "Decision Tree",
                   "Random Forest",
                   "Support Vector Machine",
                   "kNN", "Naive Bayes",
                   "Neural Network")
  
  # Loop through the list and populate the summary dataframe
  for (i in seq_along(classification_reports_combined)) {
    report <- classification_reports_combined[[i]]
    
    # Assuming each report is either a dataframe or a named vector with metrics
    print(paste("Selecting report:", report))
    new_row <- data.frame(
      Model = model_names[i],
      Accuracy = report$Accuracy,
      Precision = report$Precision,
      Recall = report$Recall,
      F1_Score = report$F1_Score
    )
    
    # Append this row to the summary dataframe
    summary_df <- rbind(summary_df, new_row)
  }
  
  # View the summary dataframe
  print(summary_df)
  # Creating a table of the model metrics
  pander(summary_df, style = "rmarkdown")
  
  # Create visualizations for comparison
  
  # Plotting Accuracy
  accuracy_plot <- ggplot(summary_df, aes(x = Model, y = Accuracy, fill = Model)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Accuracy Comparison", y = "Accuracy")
  print(accuracy_plot)
  
  # Plotting Precision
  precision_plot <- ggplot(summary_df, aes(x = Model, y = Precision, fill = Model)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Precision Comparison", y = "Precision")
  print(precision_plot)
  
  # Plotting Recall
  recall_plot <- ggplot(summary_df, aes(x = Model, y = Recall, fill = Model)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Recall Comparison", y = "Recall")
  print(recall_plot)
  
  # Plotting F1 Score
  F1_plot <- ggplot(summary_df, aes(x = Model, y = F1_Score, fill = Model)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "F1 Score Comparison", y = "F1 Score")
  print(F1_plot)
  
  
  print("ending main")
} #endof main()

# ************************************************

# clears the console area
cat("\014")

# Loads the libraries
library(dplyr)

library(pacman)
pacman::p_load(char=MYLIBRARIES,install=TRUE,character.only=TRUE)

set.seed(123)

print("Starting the Employee Attrition Analysis")

# ************************************************

main()

print("end")

