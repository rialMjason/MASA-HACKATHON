# R Code: Select Best Predictive Indicators for Malaysia's Total GHG Emissions per Capita
# Using all possible combinations with model selection criteria

# Install required packages if needed
# install.packages(c("tidyverse", "leaps", "caret", "car"))

library(tidyverse)
library(leaps)
library(caret)
library(car)

# ============================================================================
# Step 0: Load and prepare data
# ============================================================================

# Load the World Bank WDI data
df <- read.csv("WB_WDI_WIDEF.csv", stringsAsFactors = FALSE)

# Filter for Malaysia only
malaysia_df <- df %>%
  filter(Country.Name == "Malaysia") %>%
  select(-Country.Name, -Country.Code)

# Transpose to get indicators as rows and years as columns
malaysia_t <- t(malaysia_df)
colnames(malaysia_t) <- malaysia_t[1, ]
malaysia_t <- malaysia_t[-1, ]

# Convert to data frame and year columns to numeric
malaysia_clean <- as.data.frame(malaysia_t, stringsAsFactors = FALSE) %>%
  mutate(across(everything(), function(x) as.numeric(x)))

# Add year index
years <- rownames(malaysia_clean)
malaysia_clean$Year <- years

# Identify target variable: Total greenhouse gas emissions excluding LULUCF per capita
target_col <- "Total greenhouse gas emissions excluding LULUCF per capita (t CO2e/capita)"

# Check if target exists
if (!target_col %in% colnames(malaysia_clean)) {
  cat("Target column not found. Available columns:\n")
  print(colnames(malaysia_clean))
  stop("Target column not found in data")
}

# Remove rows with missing target
malaysia_clean <- malaysia_clean %>%
  filter(!is.na(!!sym(target_col)))

# Get target variable
target <- malaysia_clean[[target_col]]

# Create feature matrix (exclude target, year, and highly correlated GHG variables)
exclude_cols <- c("Year", 
                  target_col,
                  "Carbon dioxide (CO2) emissions excluding LULUCF per capita (t CO2e/capita)",
                  "Total greenhouse gas emissions excluding LULUCF (Mt CO2e)",
                  "Carbon dioxide (CO2) emissions (total) excluding LULUCF (Mt CO2e)",
                  "Total greenhouse gas emissions excluding LULUCF (% change from 1990)")

feature_cols <- colnames(malaysia_clean)[!(colnames(malaysia_clean) %in% exclude_cols)]

# Create feature data with complete cases
data_model <- data.frame(
  target = target,
  malaysia_clean[, feature_cols]
)

# Remove rows with any NA values
data_model <- data_model %>% drop_na()

print(paste("Final dataset: ", nrow(data_model), "observations,", ncol(data_model)-1, "features"))
print(paste("Target variable: ", target_col))

# ============================================================================
# Step 1: Functions to calculate CV, AIC, AICc, and BIC
# ============================================================================

calculate_metrics <- function(model, data, y_col = "target", k_folds = 5) {
  # Extract residual sum of squares
  rss <- sum(residuals(model)^2)
  n <- nrow(data)
  p <- length(coef(model)) - 1  # exclude intercept
  
  # AIC = 2p + n*ln(RSS/n)
  aic <- 2*p + n*log(rss/n)
  
  # AICc = AIC + 2p(p+1)/(n-p-1) (corrected for small sample)
  aicc <- aic + (2*p*(p+1))/(n-p-1)
  
  # BIC = p*ln(n) + n*ln(RSS/n)
  bic <- p*log(n) + n*log(rss/n)
  
  # K-Fold Cross-Validation
  set.seed(123)
  folds <- createFolds(data[[y_col]], k = k_folds, list = TRUE, returnTrain = FALSE)
  
  cv_errors <- sapply(folds, function(test_idx) {
    train_data <- data[-test_idx, ]
    test_data <- data[test_idx, ]
    
    # Refit model on training data
    model_formula <- formula(model)
    train_model <- lm(model_formula, data = train_data)
    
    # Predict on test data
    pred <- predict(train_model, newdata = test_data)
    actual <- test_data[[y_col]]
    
    # Mean squared error on test fold
    mse <- mean((pred - actual)^2)
    return(mse)
  })
  
  cv_rmse <- sqrt(mean(cv_errors))
  
  return(data.frame(
    CV_RMSE = cv_rmse,
    AIC = aic,
    AICc = aicc,
    BIC = bic,
    RSS = rss,
    R_squared = summary(model)$r.squared,
    stringsAsFactors = FALSE
  ))
}

# ============================================================================
# Step 2: All possible subsets regression
# ============================================================================

# Use regsubsets to find best models of each size
cat("\n=== Running All Possible Subsets Regression ===\n")

# Remove target column from features
X <- data_model[, -1, drop = FALSE]
y <- data_model$target

# All subsets regression
subsets <- regsubsets(x = X, y = y, nvmax = min(15, ncol(X)))

# Get results
subset_summary <- summary(subsets)

cat("Models by number of variables:\n")
print(subset_summary$which)

# ============================================================================
# Step 3: Evaluate each size with model selection criteria
# ============================================================================

results_list <- list()
model_formulas <- list()

for (size in 1:nrow(subset_summary$which)) {
  # Get feature names for this model size
  selected_features <- names(subset_summary$which[size, subset_summary$which[size, ] == TRUE])
  
  if (length(selected_features) == 0) next
  
  # Build model formula
  formula_str <- paste("target ~", paste(selected_features, collapse = " + "))
  model_formula <- as.formula(formula_str)
  
  # Fit model
  model <- lm(model_formula, data = data_model)
  
  # Calculate metrics
  metrics <- calculate_metrics(model, data_model)
  
  # Add model size information
  metrics$n_predictors <- length(selected_features)
  metrics$predictors <- paste(selected_features, collapse = ", ")
  
  results_list[[size]] <- metrics
  model_formulas[[size]] <- formula_str
}

# Combine results
results_df <- bind_rows(results_list) %>%
  arrange(n_predictors)

# Display results
results_display <- results_df %>%
  select(n_predictors, CV_RMSE, AIC, AICc, BIC, R_squared, predictors)

cat("\n=== MODEL SELECTION RESULTS ===\n")
print(results_display, n = Inf)

# ============================================================================
# Step 4: Identify best models by each criterion
# ============================================================================

cat("\n=== BEST MODELS BY CRITERION ===\n")

best_cv <- results_df %>% slice_min(CV_RMSE, n = 1)
best_aic <- results_df %>% slice_min(AIC, n = 1)
best_aicc <- results_df %>% slice_min(AICc, n = 1)
best_bic <- results_df %>% slice_min(BIC, n = 1)

cat("\nBest by CV-RMSE:\n")
print(best_cv %>% select(n_predictors, CV_RMSE, AIC, AICc, BIC, R_squared, predictors))

cat("\n\nBest by AIC:\n")
print(best_aic %>% select(n_predictors, CV_RMSE, AIC, AICc, BIC, R_squared, predictors))

cat("\n\nBest by AICc:\n")
print(best_aicc %>% select(n_predictors, CV_RMSE, AIC, AICc, BIC, R_squared, predictors))

cat("\n\nBest by BIC (favors simpler models):\n")
print(best_bic %>% select(n_predictors, CV_RMSE, AIC, AICc, BIC, R_squared, predictors))

# ============================================================================
# Step 5: Determine consensus best model
# ============================================================================

cat("\n\n=== RECOMMENDED BEST MODEL ===\n")

# BIC is recommended for variable selection as it penalizes complexity more
# Let's also look at a balance between all criteria

# Score each model (lower is better, inverted for comparison)
best_model_idx <- which.min(results_df$BIC)
best_model <- results_df[best_model_idx, ]

cat("BEST COMBINATION (using BIC - balances fit and complexity):\n\n")
cat("Number of Predictors:", best_model$n_predictors, "\n")
cat("Predictors:\n")
cat(gsub(", ", "\n  - ", paste("  - ", best_model$predictors)), "\n\n")
cat("Model Performance Metrics:\n")
cat("  - Cross-Validation RMSE:", round(best_model$CV_RMSE, 4), "\n")
cat("  - AIC:", round(best_model$AIC, 2), "\n")
cat("  - AICc:", round(best_model$AICc, 2), "\n")
cat("  - BIC:", round(best_model$BIC, 2), "\n")
cat("  - R-squared:", round(best_model$R_squared, 4), "\n")

# ============================================================================
# Step 6: Fit and display final model summary
# ============================================================================

# Extract selected features for best model
best_features <- unlist(strsplit(best_model$predictors, ", "))

final_formula <- as.formula(paste("target ~", paste(best_features, collapse = " + ")))
final_model <- lm(final_formula, data = data_model)

cat("\n\n=== FINAL MODEL SUMMARY ===\n")
print(summary(final_model))

# ============================================================================
# Step 7: Model diagnostics and visualization
# ============================================================================

par(mfrow = c(2, 2))
plot(final_model)
par(mfrow = c(1, 1))

# ============================================================================
# Step 8: Create comparison table for top 5 models
# ============================================================================

cat("\n\n=== TOP 5 MODELS BY BIC ===\n")
top_models <- results_df %>%
  arrange(BIC) %>%
  head(5) %>%
  select(n_predictors, CV_RMSE, AIC, AICc, BIC, R_squared)

print(top_models, n = Inf)

cat("\n\nFull predictor lists for top 5:\n")
top_5_full <- results_df %>%
  arrange(BIC) %>%
  head(5) %>%
  select(n_predictors, predictors)

for(i in 1:nrow(top_5_full)) {
  cat("\nModel", i, "-", top_5_full$n_predictors[i], "predictors:\n")
  cat(gsub(", ", "\n  ", paste("  ", top_5_full$predictors[i])), "\n")
}

cat("\n=== ANALYSIS COMPLETE ===\n")

