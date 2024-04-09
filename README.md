##Intro:<br />
Notice: Currently this Repo only supports daily freq time series data prediction<br />
1. Utils: Basic Function
2. cout_covid_dates_full: Automatic Anomaly Detection (Typically Anomaly means when covid, but as long as an event cause significant and long-lasting negative effects to TS, it can be detected)
3. new_prophet: Simply Wrapping Up Prophet in order to make it more convenient to use
4. calibration: Enhance the forecast precision of prophet based on practical experience (of what the model weak at)
5. Validation: Validate model using back test (similar to cross validation in tradition ML)
<br />
##Input Data<br />
The Input data should be in format like this:<br />
dt  |   y1   |   y2   |   c1   |   c2   | ...<br />
Must: dt means date. Right now this repo only supports daily freq data.<br />
Must: y1, y2, ... yn means the feature(s)(etc. y) you want to predict. Notice Prophet Only supports Univarite TS forecast, which means you can only predict one feature each time.<br />
Optional: c1, c2, ... cn means covariate(s) which can be used to enhance the prediction of y if c(s) strongly correlates with y.<br />
<br />
The names of the columns of the Input data are not necessarily the same as the example above, but before feeding it to prophet, it should be transformed into something that can be recognized by prophet:<br />
ds  |   y    |   y2   |   c1   |   c2   | ...<br />
That is: the date should be "ds", the target feature should be "y", and you should tell the models which are the covariates (if any).<br />
