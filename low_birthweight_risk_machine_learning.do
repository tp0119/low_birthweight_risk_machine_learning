*===============================================================================
* FILE: low_birthweight_risk_machine_learning.do
* PURPOSE: Econ50 Empirical Project 4: Predicting Low Birthweight Risk using Machine Learning - Part 2: Building Prediction Models for Low Birthweight Child Births
* AUTHOR: Tulsi Patel
* DATE: April 29, 2020
*===============================================================================


*-------------------------------------------------------------------------------
* Load data from natality.dta
*-------------------------------------------------------------------------------

* Change Stata working directory
cd "/Users/tulsipatel/Desktop/"

* Prepare a log file
cap log close
log using "low_birthweight_risk_machine_learning", replace

* Load in cleaned natality data 
use natality, clear

* Install binscatter
ssc install binscatter, replace

* Set seed
set seed 123


*-------------------------------------------------------------------------------
* Section 1
*-------------------------------------------------------------------------------

* Part a: number of observations in training and test sample
sum birthweight if training == 1
sum birthweight if training == 0

* Part b: balance table for random assignment to training and test samples

* mean and standard deviation for babies in test group
sum baby_female if training == 1
sum mom_age if training == 1
sum mom_race_white if training == 1
sum mom_race_black if training == 1
sum mom_use_alcohol if training == 1
sum mom_yrs_educ if training == 1
sum mom_previous_preterm if training == 1

* differences between training and test means & standard error for the differences in means
regress baby_female training, r 
regress mom_age training, r 
regress mom_race_white training, r 
regress mom_race_black training, r
regress mom_use_alcohol training, r 
regress mom_yrs_educ training, r
regress mom_previous_preterm training, r  


*-------------------------------------------------------------------------------
* Section 2
*-------------------------------------------------------------------------------

* Part a: multiple regression model to predict birthweight
regress birthweight mom_age mom_race_black mom_use_alcohol mom_yrs_educ if training == 1

* get predictions for all observations (both training and test samples)
predict yhat_reg 

* Part b: calculate squared prediction errors
gen squared_error_reg = (birthweight - yhat_reg)^2

* mean squared prediction errors for training data
sum squared_error_reg if training == 1

* display the square root of the mean squared error
display sqrt(r(mean))

* mean squared prediction errors for test data
sum squared_error_reg if training == 0 
display sqrt(r(mean))


*-------------------------------------------------------------------------------
* Section 3
*-------------------------------------------------------------------------------

* Part a: decision tree with max depth = 25
pytree birthweight mom_age mom_race_black mom_use_alcohol mom_yrs_educ if training == 1, type(regress) max_depth(25)

* get predictions for all observations
predict yhat_tree

* Part b: mean squared error in training data
gen squared_error_tree = (birthweight - yhat_tree)^2
sum squared_error_tree if training == 1
display sqrt(r(mean))

* mean squared error in test data
sum squared_error_tree if training == 0
display sqrt(r(mean))

* Part c: cross validation

* divide training data into 5 folds
gen fold = ceil(runiform()*5) if training == 1

* generate variables rmse_depth_1-rmse_depth_15 to store RMSE associated with trees of depth 1-15
qui forval i=1/15 {
    gen rmse_depth_`i' = .
}

* cross-validation: loop over each fold i = 1, 2, 3, 4, 5
forval i=1/5 {
	
	* loop over max tree depth j = 1, 2, ..., 15
	qui forval j=1/15 {
	    
		* run decision tree with max depth j on all folds except for fold i
		pytree birthweight mom_age mom_race_black mom_use_alcohol mom_yrs_educ if fold!=`i', type(regress) max_depth(`j')
		
		* generate RMSE for observations in fold i (out-of-sample)
		predict yhat if fold==`i', xb
		generate squared_error = (birthweight - yhat)^2
		sum squared_error
		local rmse = sqrt(`r(mean)')
		
		* save RMSE in a variable
		replace rmse_depth_`j' = `rmse' in `i'
		
		* drop prediction for next iteration in loop
		drop yhat squared_error
	}
	
}

* display cross-validated mean RMSEs for each tree depth, taking mean across folds
sum rmse_depth*

* Part d: run decision tree with tree depth 3
pytree birthweight mom_age mom_race_black mom_use_alcohol mom_yrs_educ if training == 1, type(regress) max_depth(3)

* save predictions
predict yhat_tree_3

* RSME in training data
gen squared_error_tree_3 = (birthweight - yhat_tree_3)^2
sum squared_error_tree_3 if training == 1
display sqrt(r(mean))

* RSME in test data
sum squared_error_tree_3 if training == 0
display sqrt(r(mean))


*-------------------------------------------------------------------------------
* Section 4
*-------------------------------------------------------------------------------

* Part a: generate random forest with 100 trees and max tree depth = 10
pyforest birthweight mom_age mom_race_black mom_use_alcohol mom_yrs_educ if training == 1, type(regress) n_estimators(100) max_depth(10)

* save predictions
predict yhat_forest

* Part b: RMSE for random forest

* RSME in training data
gen squared_error_forest = (birthweight - yhat_forest)^2
sum squared_error_forest if training == 1
display sqrt(r(mean))

* RSME in test data
sum squared_error_forest if training == 0
display sqrt(r(mean))


*-------------------------------------------------------------------------------
* Section 5
*-------------------------------------------------------------------------------

* Part b: create low birthweight indicator variable (same as Part 1)
gen low_bw = 0
replace low_bw = 1 if birthweight < 2500

* binned scatter plot of relationship between low_bw and random forest model predictions on the test sample
binscatter low_bw yhat_forest if training == 0
graph export lowbw_vs_forest.png, replace

* Part c: experimentation with random forest model
* (1) tree depth of 3 (optimal depth as found in Section 3c)
pyforest birthweight mom_age mom_race_black mom_use_alcohol mom_yrs_educ if training == 1, type(regress) n_estimators(100) max_depth(3)

predict yhat_forest_3

gen squared_error_forest_3 = (birthweight - yhat_forest_3)^2
sum squared_error_forest_3 if training == 0
display sqrt(r(mean))

* (2) tree depth of 3 and using 1,000 trees 
pyforest birthweight mom_age mom_race_black mom_use_alcohol mom_yrs_educ if training == 1, type(regress) n_estimators(1000) max_depth(3)

predict yhat_forest_1000_3

gen squared_error_forest_1000_3 = (birthweight - yhat_forest_1000_3)^2
sum squared_error_forest_1000_3 if training == 0
display sqrt(r(mean))


*-------------------------------------------------------------------------------
* Section 6
*-------------------------------------------------------------------------------

* Part b: using the ranfom forest model to compute test sample RMSE for children with black vs. non-black mothers

* RMSE for babies with black mothers 
sum squared_error_forest if training == 0 & mom_race_black == 1
display sqrt(r(mean))

* RMSE for babies with non-black mothers 
sum squared_error_forest if training == 0 & mom_race_black == 0
display sqrt(r(mean))


