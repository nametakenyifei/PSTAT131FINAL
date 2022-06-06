library(ggplot2)
library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggthemes)
library(discrim)
library(poissonreg)
library(corrr)
library(klaR)
library(ISLR)
library(ISLR2)
library(purrr)
library(janitor)
library(psych) 
library(randomForest)
library(xgboost)
library(rpart.plot)
library(ranger)
library(vip)
library(pROC)
tidymodels_prefer()

loldata <- read_csv("DATA/high_diamond_ranked_10min.csv")

# save the cleaned data
lol <- clean_names(loldata)

lol_blue <-  lol[ , 0:21]

important <- lol_blue[c( "blue_wins", "blue_first_blood", 
                         "blue_kills",  "blue_deaths",
                         "blue_assists","blue_elite_monsters", 
                         "blue_dragons", "blue_total_gold", 
                         "blue_avg_level", "blue_total_experience", 
                         "blue_gold_diff", "blue_experience_diff", 
                         "blue_total_minions_killed")]

set.seed(2022)

important <- important  %>% 
  mutate(blue_wins = factor(blue_wins, 
                            levels = c(0, 1)),
         blue_first_blood = factor(blue_first_blood),
  )

lol_split <- important %>% 
  initial_split(strata = blue_wins, prop = 0.75)
lol_train <- training(lol_split)
lol_test <- testing(lol_split)

dim(lol_train)

lol_recipe <- recipe(blue_wins ~ ., data = lol_train) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_predictors())

lol_folds <- vfold_cv(lol_train, v = 5, strata = 'blue_wins')

control <- control_resamples(save_pred = TRUE)

tree_spec <- decision_tree() %>%
  set_engine("rpart")

class_tree_spec <- tree_spec %>%
  set_mode("classification")

class_tree_fit <- class_tree_spec %>%
  fit(blue_wins ~ .,
      data = lol_train)

class_tree_wf <- workflow() %>%
  add_model(class_tree_spec %>% set_args(cost_complexity = tune())) %>%
  add_recipe(lol_recipe)

param_grid <- grid_regular(cost_complexity(range = c(-5, -1)), levels = 5)

tune_tree_res <- tune_grid(
  class_tree_wf, 
  resamples = lol_folds, 
  grid = param_grid, 
  metrics = metric_set(roc_auc))

write_rds(tune_tree_res, "model_results/dt_tune.rds")

forest_spec <- rand_forest(
  mode = "classification",
  mtry = tune(),
  trees = tune(),
  min_n = tune()
)%>%
  set_engine("ranger", importance = "impurity") 

forest_wf <- workflow() %>%
  add_model(forest_spec %>% 
              set_args(mtry = tune(), trees = tune(),
                       min_n = tune()
              )
  ) %>%
  add_recipe(lol_recipe)

forest_grid <- grid_regular(mtry(range = c(1, 12)), 
                            trees(range = c(1, 5)),
                            min_n(range = c(1, 5)),
                            levels = 5)
forest_grid

forest_tune_res <- tune_grid(
  forest_wf, 
  resamples = lol_folds, 
  grid = forest_grid, 
  metrics = metric_set(roc_auc)
)

write_rds(forest_tune_res, "model_results/forest_tune.rds")

boost_spec <- boost_tree(
  mode = "classification",
  engine = "xgboost",
  trees = tune(),
)
boost_wf <- workflow() %>%
  add_model(boost_spec %>% 
              set_args(trees = tune()
              )
  ) %>%
  add_recipe(lol_recipe)

boost_grid <- grid_regular(trees(range = c(10, 200)), 
                           levels = 10)

boost_tune_res <- tune_grid(
  boost_wf, 
  resamples = lol_folds, 
  grid = boost_grid, 
  metrics = metric_set(roc_auc)
)

write_rds(boost_tune_res, "model_results/bt_tune.rds")

elastic_net_spec <- multinom_reg(penalty = tune(), 
                                 mixture = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("glmnet")

en_workflow <- workflow() %>% 
  add_recipe(lol_recipe) %>% 
  add_model(elastic_net_spec)

en_grid <- grid_regular(penalty(range = c(-5, 5)), 
                        mixture(range = c(0, 1)), levels = 10)

tune_res <- tune_grid(
  en_workflow,
  resamples = lol_folds, 
  grid = en_grid
)

write_rds(tune_res, file = "model_results/en.rds")
