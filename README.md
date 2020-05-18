# Seattle-AirBnb-Listings

For this project, I was interested in seeing which variables are the most important in determining a rate for a unit as well as creating model(s) that would accurately predict a rate using those variables (# of reviews, # of bedrooms, # of bathrooms and room type). The models that I employed are linear regression, ridge regression, lasso, random forest and boosting. The dataset (CSV file) and R code are included in this repository as well.

First and foremost, before creating any models I performed exploratory data analysis on the variables that I was going to use in my models. I first performed a correlation matrix for my variables. Looking at the matrix, there are a couple og high correlations (between bedrooms and bathrooms for instance, which makes sense). However, looking at the scatterplots of the variables, there look to be no specific correlations, and therefore no interaction terms were added in the linear regression. The only notable relationship between variables in the scatterplots is that there is a non-linear relationship between price and reviews, which is why the square of reviews is added in the linear regession model.

Upon performing the first regression model (Rate), all of the variables are significant. Looking at the diagnostic plots for this model, the normal distribution plot is not the best. For this reason, I create a second linear regression model with the log of price as the response in order to gain a better fit for the model (Rate2). Again, all of the variables are significant. Looking at the normal distribution plot, it is a much better fit for this second model.

Performing ridge regression, using cross-validation we obtain a lambda of 4.94 and the coefficients show that room type plays a huge role in determining price, followed by both bedrooms and bathrooms not too far behind. Reviews play the least important rule. Performing lasso, we get a lambda of .186. Given that this is pretty low, it does not deviate too far from a standard linear regression model. We can similar results in terms of coefficients as compared to the ridge regression. These two models have similar test mean squared errors (MSEs) as well.

Onto the random forest, we see that bedrooms and bathrooms play the most important role, followed by room type. The model performs slightly better than the ridge regression and lasso, but not by much. As for the boosting, it shows that reviews actually play the biggest role, however the test MSE shows that boosting performs the worst out of all the models.
