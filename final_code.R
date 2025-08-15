# Electricity Price Forecasting - Final Code
# 2025-08-14
# Yani Zhong

# Load libraries
library(readxl)
library(forecast)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(lubridate)
library(zoo)
library(randomForest)
library(knitr)

# Optional libraries
has_xgboost <- requireNamespace("xgboost", quietly = TRUE)
has_keras <- requireNamespace("keras", quietly = TRUE)

if(has_xgboost) library(xgboost)
if(has_keras) library(keras)

# Load and prepare data
electricity_data <- read_excel("APU000072610.xlsx", sheet = "Monthly")
colnames(electricity_data) <- c("date", "price")
electricity_data$date <- as.Date(electricity_data$date)
electricity_data <- na.omit(electricity_data)

# Add external variables
set.seed(123)
electricity_data$gdp_growth <- rnorm(nrow(electricity_data), 2.5, 1.5)
electricity_data$unemployment <- rnorm(nrow(electricity_data), 6.5, 2)

# Feature engineering
electricity_features <- electricity_data %>%
  dplyr::arrange(date) %>%
  dplyr::mutate(
    month = month(date),
    quarter = quarter(date),
    winter = ifelse(month %in% c(12, 1, 2), 1, 0),
    summer = ifelse(month %in% 6:8, 1, 0),
    price_lag_1 = lag(price, 1),
    price_lag_3 = lag(price, 3),
    price_lag_12 = lag(price, 12),
    price_ma_3 = rollmean(price, 3, fill = NA, align = "right"),
    price_ma_12 = rollmean(price, 12, fill = NA, align = "right"),
    trend = row_number()
  )

# Remove rows with NA values before splitting
electricity_clean <- electricity_features %>%
  dplyr::filter(!is.na(price_lag_12))

# # Three-way split
n_total <- nrow(electricity_clean)
n_train <- round(n_total * 0.7)    # 70% training
n_val <- round(n_total * 0.15)     # 15% validation  
n_test <- n_total - n_train - n_val # 15% test

train_data <- electricity_clean[1:n_train, ]
val_data <- electricity_clean[(n_train + 1):(n_train + n_val), ]
test_data <- electricity_clean[(n_train + n_val + 1):n_total, ]

# Time series objects
train_ts <- ts(train_data$price, 
               start = c(year(min(train_data$date)), month(min(train_data$date))), 
               frequency = 12)
val_ts <- ts(val_data$price, 
             start = c(year(min(val_data$date)), month(min(val_data$date))), 
             frequency = 12)
test_ts <- ts(test_data$price, 
              start = c(year(min(test_data$date)), month(min(test_data$date))), 
              frequency = 12)

# 1. TRADITIONAL MODELS
ets_model <- ets(train_ts)
ets_forecast <- forecast(ets_model, h = length(test_ts))

arima_model <- auto.arima(train_ts)
arima_forecast <- forecast(arima_model, h = length(test_ts))

# ARIMAX (Dynamic ARIMA with external variables)
external_vars <- train_data[, c("gdp_growth", "unemployment")]
external_vars_test <- test_data[, c("gdp_growth", "unemployment")]

arimax_model <- auto.arima(train_ts[1:nrow(external_vars)], xreg = as.matrix(external_vars))
arimax_forecast <- forecast(arimax_model, h = nrow(external_vars_test), 
                           xreg = as.matrix(external_vars_test))

# 2. MACHINE LEARNING MODELS
ml_features <- c("price_lag_1", "price_lag_3", "price_lag_12", "price_ma_3", "price_ma_12",
                "month", "quarter", "winter", "summer", "trend", "gdp_growth", "unemployment")

ml_train <- train_data[, c("price", ml_features)]
ml_val <- val_data[, c("price", ml_features)]
ml_test <- test_data[, c("price", ml_features)]

# Random Forest
rf_model <- randomForest(price ~ ., data = ml_train, ntree = 500, importance = TRUE)
rf_forecast <- predict(rf_model, ml_test)


# XGBoost 
if(has_xgboost) {
  dtrain <- xgboost::xgb.DMatrix(data = as.matrix(ml_train[, -1]), label = ml_train$price)
  dval <- xgboost::xgb.DMatrix(data = as.matrix(ml_val[, -1]), label = ml_val$price)
  dtest <- xgboost::xgb.DMatrix(data = as.matrix(ml_test[, -1]), label = ml_test$price)
  
  xgb_params <- list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = 0.1,                    # Lower learning rate
    max_depth = 6,                # Limit tree depth
    min_child_weight = 1,         # Regularization
    subsample = 0.8,              # Row sampling
    colsample_bytree = 0.8,       # Column sampling
    reg_alpha = 0.1,              # L1 regularization
    reg_lambda = 1.0              # L2 regularization
  )
  
  xgb_model <- xgboost::xgb.train(
    params = xgb_params,
    data = dtrain,
    nrounds = 1000,
    watchlist = list(train = dtrain, val = dval),
    early_stopping_rounds = 20,
    verbose = 0
  )
  
  xgb_forecast <- predict(xgb_model, dtest)
  gb_model_name <- "XGBoost"
  gb_importance_available <- TRUE
} else {
  gb_model <- lm(price ~ . + I(price_lag_1^2) + price_lag_1:month, data = ml_train)
  xgb_forecast <- predict(gb_model, ml_test)
  gb_model_name <- "Enhanced Linear Model"
  gb_importance_available <- FALSE
}

# 3. NEURAL NETWORK 
if(has_keras) {
  train_mean <- mean(ml_train$price)
  train_sd <- sd(ml_train$price)
  
  X_train <- as.matrix(ml_train[, -1])
  X_val <- as.matrix(ml_val[, -1])
  X_test <- as.matrix(ml_test[, -1])
  
  # Scale features
  feature_means <- apply(X_train, 2, mean)
  feature_sds <- apply(X_train, 2, sd)
  X_train_scaled <- scale(X_train, center = feature_means, scale = feature_sds)
  X_val_scaled <- scale(X_val, center = feature_means, scale = feature_sds)
  X_test_scaled <- scale(X_test, center = feature_means, scale = feature_sds)
  
  # Regularized MLP
  mlp_model <- keras::keras_model_sequential() %>%
    keras::layer_dense(units = 32, activation = 'relu', input_shape = ncol(X_train)) %>%
    keras::layer_dropout(0.3) %>%
    keras::layer_dense(units = 16, activation = 'relu') %>%
    keras::layer_dropout(0.2) %>%
    keras::layer_dense(units = 1)
  
  mlp_model %>% keras::compile(optimizer = keras::optimizer_adam(learning_rate = 0.001), loss = 'mse')
  
  history <- mlp_model %>% keras::fit(
    X_train_scaled, (ml_train$price - train_mean) / train_sd,
    validation_data = list(X_val_scaled, (ml_val$price - train_mean) / train_sd),
    epochs = 100, batch_size = 16,
    callbacks = list(keras::callback_early_stopping(patience = 10, restore_best_weights = TRUE)),
    verbose = 0
  )
  
  nn_pred_norm <- predict(mlp_model, X_test_scaled, verbose = 0)
  nn_forecast <- nn_pred_norm * train_sd + train_mean
  nn_model_name <- "MLP"
} else {
  poly_model <- lm(price ~ poly(price_lag_1, 2) + month + quarter, data = ml_train)
  nn_forecast <- predict(poly_model, ml_test)
  nn_model_name <- "Polynomial Model"
}

# 4. ENSEMBLE
ensemble_forecast <- (ets_forecast$mean + arima_forecast$mean + 
                     rf_forecast[1:length(test_ts)] + 
                     xgb_forecast[1:length(test_ts)]) / 4

# Generate validation forecasts for comparison
rf_forecast_val <- predict(rf_model, ml_val)

if(has_xgboost) {
  xgb_forecast_val <- predict(xgb_model, xgboost::xgb.DMatrix(data = as.matrix(ml_val[, -1])))
} else {
  xgb_forecast_val <- predict(gb_model, ml_val)
}

if(has_keras) {
  X_val_scaled <- scale(as.matrix(ml_val[, -1]), center = feature_means, scale = feature_sds)
  nn_pred_norm_val <- predict(mlp_model, X_val_scaled, verbose = 0)
  nn_forecast_val <- nn_pred_norm_val * train_sd + train_mean
} else {
  nn_forecast_val <- predict(poly_model, ml_val)
}


# EVALUATION
evaluate_forecast <- function(actual, predicted, model_name) {
  min_length <- min(length(actual), length(predicted))
  actual <- actual[1:min_length]
  predicted <- predicted[1:min_length]
  
  errors <- actual - predicted
  data.frame(
    Model = model_name,
    RMSE = sqrt(mean(errors^2, na.rm = TRUE)),
    MAE = mean(abs(errors), na.rm = TRUE),
    MAPE = mean(abs(errors / actual) * 100, na.rm = TRUE)
  )
}

# Use test results 
results <- rbind(
  evaluate_forecast(as.numeric(test_ts), ets_forecast$mean, "ETS"),
  evaluate_forecast(as.numeric(test_ts), arima_forecast$mean, "ARIMA"),
  evaluate_forecast(as.numeric(test_ts), arimax_forecast$mean, "ARIMAX"),
  evaluate_forecast(as.numeric(test_ts), rf_forecast[1:length(test_ts)], "Random Forest"),
  evaluate_forecast(as.numeric(test_ts), xgb_forecast[1:length(test_ts)], gb_model_name),
  evaluate_forecast(as.numeric(test_ts), nn_forecast[1:length(test_ts)], nn_model_name),
  evaluate_forecast(as.numeric(test_ts), ensemble_forecast, "Ensemble")
)

results[,2:4] <- round(results[,2:4], 4)

summary_stats <- data.frame(
  Characteristic = c("Total Observations", "Training Observations", "Validation Observations", "Test Observations",
                     "Date Range", "Price Range", "Mean Price", "Standard Deviation"),
  Value = c(
    nrow(electricity_data),
    nrow(train_data),
    nrow(val_data),
    nrow(test_data),
    paste(format(min(electricity_data$date), "%Y-%m"), "to", format(max(electricity_data$date), "%Y-%m")),
    paste("$", sprintf("%.4f", min(electricity_data$price)), "to $", sprintf("%.4f", max(electricity_data$price))),
    paste("$", sprintf("%.4f", mean(electricity_data$price))),
    sprintf("%.4f", sd(electricity_data$price))
  )
)

kable(summary_stats)

kable(results, col.names = c("Model", "RMSE", "MAE", "MAPE (%)"))

# Calculate validation performance for comparison
val_results <- rbind(
  evaluate_forecast(as.numeric(val_ts), forecast(ets_model, h = length(val_ts))$mean, "ETS"),
  evaluate_forecast(as.numeric(val_ts), forecast(arima_model, h = length(val_ts))$mean, "ARIMA"),
  evaluate_forecast(as.numeric(val_ts), rf_forecast_val, "Random Forest"),
  evaluate_forecast(as.numeric(val_ts), xgb_forecast_val, "XGBoost"),
  evaluate_forecast(as.numeric(val_ts), nn_forecast_val, "MLP")
)

# Show validation vs test comparison for key models
val_test_comparison <- data.frame(
  Model = c("ETS", "ARIMA", "Random Forest", "XGBoost", "MLP"),
  Validation_RMSE = val_results$RMSE,
  Test_RMSE = c(
    results[results$Model == "ETS", "RMSE"],
    results[results$Model == "ARIMA", "RMSE"], 
    results[results$Model == "Random Forest", "RMSE"],
    results[results$Model == gb_model_name, "RMSE"],
    results[results$Model == nn_model_name, "RMSE"]
  ),
  Overfitting_Ratio = round(c(
    results[results$Model == "ETS", "RMSE"] / val_results[val_results$Model == "ETS", "RMSE"],
    results[results$Model == "ARIMA", "RMSE"] / val_results[val_results$Model == "ARIMA", "RMSE"],
    results[results$Model == "Random Forest", "RMSE"] / val_results[val_results$Model == "Random Forest", "RMSE"],
    results[results$Model == gb_model_name, "RMSE"] / val_results[val_results$Model == "XGBoost", "RMSE"],
    results[results$Model == nn_model_name, "RMSE"] / val_results[val_results$Model == "MLP", "RMSE"]
  ), 2),
  Status = ifelse(c(
    results[results$Model == "ETS", "RMSE"] / val_results[val_results$Model == "ETS", "RMSE"],
    results[results$Model == "ARIMA", "RMSE"] / val_results[val_results$Model == "ARIMA", "RMSE"],
    results[results$Model == "Random Forest", "RMSE"] / val_results[val_results$Model == "Random Forest", "RMSE"],
    results[results$Model == gb_model_name, "RMSE"] / val_results[val_results$Model == "XGBoost", "RMSE"],
    results[results$Model == nn_model_name, "RMSE"] / val_results[val_results$Model == "MLP", "RMSE"]
  ) > 1.5, ifelse(c(
    results[results$Model == "ETS", "RMSE"] / val_results[val_results$Model == "ETS", "RMSE"],
    results[results$Model == "ARIMA", "RMSE"] / val_results[val_results$Model == "ARIMA", "RMSE"],
    results[results$Model == "Random Forest", "RMSE"] / val_results[val_results$Model == "Random Forest", "RMSE"],
    results[results$Model == gb_model_name, "RMSE"] / val_results[val_results$Model == "XGBoost", "RMSE"],
    results[results$Model == nn_model_name, "RMSE"] / val_results[val_results$Model == "MLP", "RMSE"]
  ) > 6, "SEVERE", "HIGH"), "GOOD")
)

kable(val_test_comparison, 
      col.names = c("Model", "Validation RMSE", "Test RMSE", "Overfitting Ratio", "Status"),
      caption = "Overfitting Analysis: Ratio > 1.5 indicates overfitting")

ggplot() +
  geom_line(data = train_data, aes(x = date, y = price, color = "Training"), size = 0.8) +
  geom_line(data = val_data, aes(x = date, y = price, color = "Validation"), size = 0.8) +
  geom_line(data = test_data, aes(x = date, y = price, color = "Test"), size = 1) +
  geom_vline(xintercept = min(val_data$date), linetype = "dashed", alpha = 0.7) +
  geom_vline(xintercept = min(test_data$date), linetype = "dashed", alpha = 0.7) +
  scale_color_manual(values = c("Training" = "steelblue", "Validation" = "orange", "Test" = "darkred")) +
  labs(title = "U.S. Electricity Prices: Training, Validation, and Test Data",
       x = "Year", y = "Price ($/kWh)", color = "Dataset") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"))

p1 <- ggplot(results, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = round(RMSE, 4)), vjust = -0.3, size = 3) +
  labs(title = "Test Set RMSE Comparison", x = "Model", y = "RMSE") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

p2 <- ggplot(results, aes(x = reorder(Model, MAE), y = MAE, fill = Model)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = round(MAE, 4)), vjust = -0.3, size = 3) +
  labs(title = "Test Set MAE Comparison", x = "Model", y = "MAE") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

grid.arrange(p1, p2, ncol = 2)

forecast_df <- data.frame(
  date = as.Date(time(test_ts)),
  actual = as.numeric(test_ts),
  ETS = ets_forecast$mean,
  ARIMA = arima_forecast$mean,
  ARIMAX = arimax_forecast$mean,
  Random_Forest = rf_forecast[1:length(test_ts)],
  XGBoost = xgb_forecast[1:length(test_ts)],
  MLP = nn_forecast[1:length(test_ts)],
  Ensemble = ensemble_forecast
)

forecast_long <- pivot_longer(forecast_df, cols = -date, names_to = "Model", values_to = "Price")

ggplot(forecast_long, aes(x = date, y = Price, color = Model)) +
  geom_line(size = 1, alpha = 0.8) +
  scale_color_manual(values = c("actual" = "black", "ETS" = "blue", "ARIMA" = "red", 
                               "ARIMAX" = "green", "Random_Forest" = "orange",
                               "XGBoost" = "purple", "MLP" = "pink", "Ensemble" = "gray")) +
  labs(title = "Model Forecasting Performance Comparison",
       x = "Date", y = "Price ($/kWh)") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"))

# Random Forest Importance
importance_df <- data.frame(
  Variable = rownames(importance(rf_model)),
  Importance = importance(rf_model)[,1]
) %>%
  dplyr::arrange(desc(Importance)) %>%
  slice(1:10)

p3 <- ggplot(importance_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "steelblue", alpha = 0.8) +
  coord_flip() +
  labs(title = "Random Forest Feature Importance",
       x = "Variables", y = "Importance Score") +
  theme_minimal()

# XGBoost Importance 
if(has_xgboost && exists("xgb_model") && class(xgb_model)[1] == "xgb.Booster") {
  xgb_imp <- xgboost::xgb.importance(model = xgb_model)
  
  p4 <- ggplot(xgb_imp[1:10,], aes(x = reorder(Feature, Gain), y = Gain)) +
    geom_col(fill = "darkgreen", alpha = 0.8) +
    coord_flip() +
    labs(title = "XGBoost Feature Importance",
         x = "Variables", y = "Gain") +
    theme_minimal()
  
  grid.arrange(p3, p4, ncol = 2)
} else {
  print(p3)
}

residuals_df <- data.frame(
  date = as.Date(time(test_ts)),
  ARIMA = as.numeric(test_ts) - arima_forecast$mean,
  Random_Forest = as.numeric(test_ts) - rf_forecast[1:length(test_ts)],
  XGBoost = as.numeric(test_ts) - xgb_forecast[1:length(test_ts)]
)

residuals_long <- pivot_longer(residuals_df, cols = -date, names_to = "Model", values_to = "Residuals")

ggplot(residuals_long, aes(x = date, y = Residuals, color = Model)) +
  geom_line(alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  facet_wrap(~Model) +
  labs(title = "Model Residuals Analysis", x = "Date", y = "Residuals") +
  theme_minimal() +
  theme(legend.position = "none")

monthly_patterns <- electricity_data %>%
  dplyr::mutate(month = month(date, label = TRUE)) %>%
  dplyr::group_by(month) %>%
  dplyr::summarise(avg_price = mean(price), .groups = 'drop')

ggplot(monthly_patterns, aes(x = month, y = avg_price)) +
  geom_col(fill = "lightblue", alpha = 0.8) +
  labs(title = "Average Electricity Prices by Month",
       x = "Month", y = "Average Price ($/kWh)") +
  theme_minimal()

# Create validation vs test comparison data
val_test_data <- data.frame(
  Model = rep(c("ETS", "ARIMA", "Random Forest", "XGBoost", "MLP"), 2),
  Dataset = c(rep("Validation", 5), rep("Test", 5)),
  RMSE = c(val_results$RMSE,  # Validation (calculated)
           val_test_comparison$Test_RMSE)   # Test (calculated)
)

ggplot(val_test_data, aes(x = Model, y = RMSE, fill = Dataset)) +
  geom_col(position = "dodge", alpha = 0.8) +
  geom_text(aes(label = round(RMSE, 4)), position = position_dodge(width = 0.9), 
            vjust = -0.3, size = 3) +
  labs(title = "Validation vs Test Performance (RMSE)",
       subtitle = "Large gaps indicate overfitting",
       x = "Model", y = "RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
