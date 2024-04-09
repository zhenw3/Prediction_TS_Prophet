//Intro:
Notice: Currently this Repo only supports daily freq time series data prediction
1. Utils: Basic Function
2. cout_covid_dates_full: Automatic Anomaly Detection (Typically Anomaly means when covid, but as long as an event cause significant and long-lasting negative effects to TS, it can be detected)
3. new_prophet: Simply Wrapping Up Prophet in order to make it more convenient to use
4. calibration: Enhance the forecast precision of prophet based on practical experience (of what the model weak at)
5. Validation: Validate model using back test (similar to cross validation in tradition ML)

//Input Data
The Input data should be in format like this:
dt  |   y1   |   y2   |   c1   |   c2   | ...
Must: dt means date. Right now this repo only supports daily freq data.
Must: y1, y2, ... yn means the feature(s)(etc. y) you want to predict. Notice Prophet Only supports Univarite TS forecast, which means you can only predict one feature each time.
Optional: c1, c2, ... cn means covariate(s) which can be used to enhance the prediction of y if c(s) strongly correlates with y.

The names of the columns of the Input data are not necessarily the same as the example above, but before feeding it to prophet, it should be transformed into something that can be recognized by prophet:
ds  |   y    |   y2   |   c1   |   c2   | ...
That is: the date should be "ds", the target feature should be "y", and you should tell the models which are the covariates (if any).
