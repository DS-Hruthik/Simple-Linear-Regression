'''
# Import necessary libraries for data manipulation, mathematical calculations, visualization, and modeling
import pandas as pd  # Data Manipulation
import numpy as np   # Mathematical calculations
import matplotlib.pyplot as plt  # Data Visualization
import seaborn as sns  # Data Visualization
import joblib  # Saving and loading model
import pickle  # Saving and loading model
from sklearn.compose import ColumnTransformer  # Column Transformer
from sklearn.pipeline import Pipeline  # Pipeline for modeling
from sklearn.impute import SimpleImputer  # Imputing missing values
from sklearn.pipeline import make_pipeline  # Pipeline for modeling
from feature_engine.outliers import Winsorizer  # Handling outliers
from sklearn.model_selection import train_test_split  # Splitting data into train and test sets
import statsmodels.formula.api as smf  # Statsmodels for statistical modeling
from sklearn.preprocessing import PolynomialFeatures  # Polynomial features for modeling
from sklearn.linear_model import LinearRegression
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import statsmodels.formula.api as smf

import joblib

wcat = pd.read_csv(r"C:\Users\hruth\Desktop\self pace learning\course\ML\l 5.c.Simple Linear Regression-20240730T090446Z-001\5.c.Simple Linear Regression\wc-at.csv")


X = wcat['Waist'] #❌
Y = wcat['AT'] #❌

X = pd.DataFrame(wcat['Waist'])  # Create DataFrame for predictor
Y = pd.DataFrame(wcat['AT'])    # Create DataFrame for target variable


numeric_features = ['Waist']

wcat.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,8))

winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables= ['Waist'])

winsor

num_pipeline = Pipeline(
    [('impute', SimpleImputer(strategy= 'mean'))])

outlier_pipeline = Pipeline(
    [('winsor', winsor)])

preprocessor = ColumnTransformer(
    [('num', num_pipeline, numeric_features)])

preprocessor1 = ColumnTransformer(
    [('wins', outlier_pipeline, numeric_features)])


impute_data = preprocessor.fit(X)
wcat['Waist'] = pd.DataFrame(impute_data.transform(X))

X2 = pd.DataFrame(wcat['Waist'])
winz_data = preprocessor.fit(X2)

wcat['Waist'] = pd.DataFrame(winz_data.transform(X))


joblib.dump(impute_data, 'imputation')


joblib.dump(winz_data, 'winzor')

wcat.plot(kind = 'box', subplots = True, sharey = False, figsize = (15,8))



################    plots

plt.figure(figsize=(10, 6))  # Set figure size for better visualization
plt.bar(height=wcat.AT, x=np.arange(1, 110, 1))  # Create bar graph with index as x-axis
plt.xlabel('Index')  # Label the x-axis
plt.ylabel('AT Value')  # Label the y-axis
plt.title('Bar Graph of AT Values')  # Add a title for clarity
plt.show()  # Display the bar graph

# 2. Histogram of Target Variable (AT)
plt.figure(figsize=(10, 6))  # Set figure size for better visualization
plt.hist(wcat.AT)  # Create a histogram of AT values
plt.xlabel('AT Value')  # Label the x-axis
plt.ylabel('Frequency')  # Label the y-axis
plt.title('Histogram of AT Values')  # Add a title for clarity
plt.show()  # Display the histogram

# 3. Bar Graph of Predictor Variable (Waist)
plt.figure(figsize=(10, 6))  # Set figure size for better visualization
plt.bar(height=wcat.Waist, x=np.arange(1, 110, 1))  # Create bar graph with index as x-axis
plt.xlabel('Index')  # Label the x-axis
plt.ylabel('Waist Circumference')  # Label the y-axis
plt.title('Bar Graph of Waist Circumference')  # Add a title for clarity
plt.show()  # Display the bar graph

# 4. Histogram of Predictor Variable (Waist)
plt.figure(figsize=(10, 6))  # Set figure size for better visualization
plt.hist(wcat.Waist)  # Create a histogram of waist circumference values
plt.xlabel('Waist Circumference')  # Label the x-axis
plt.ylabel('Frequency')  # Label the y-axis
plt.title('Histogram of Waist Circumference')  # Add a title for clarity
plt.show() 



import sweetviz as sv
report = sv.analyze(wcat)
report.show_html('EDAreport.html')
import numpy as np

# assign a built-in warning class so sweetviz can use it
setattr(np, "VisibleDeprecationWarning", DeprecationWarning)



#✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅ 
# model
# linear regression

model = smf.ols('AT ~ Waist', data = wcat).fit()
model.summary()

pred = model.predict(pd.DataFrame(wcat['Waist']))
pred

res = wcat.AT - pred
res_sqr = res * res
mse = np.mean(res_sqr)
rmse = np.sqrt(mse)
rmse

plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist, pred, "r")
plt.legend(['Observed data', 'Predicted line'])



# model Tuning with transformations
# model 2
# logarithmic regression
plt.scatter(x = np.log(wcat['Waist']), y = wcat.AT, color = 'brown')
np.corrcoef(np.log(wcat['Waist']), wcat['AT'])

model2 = smf.ols('AT ~ np.log(Waist)', data = wcat).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(wcat['Waist']))
pred2

res2 = wcat.AT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

plt.scatter(np.log(wcat['Waist']), wcat['AT'])
plt.plot(np.log(wcat['Waist']), pred2, color = 'red')
plt.legend(['Observed data', 'Predicted data'])


#model3
plt.scatter(x = wcat.Waist, y = np.log(wcat['AT']), color = 'orange')
np.corrcoef(wcat.Waist, np.log(wcat['AT']))

model3 = smf.ols('np.log(AT) ~ Waist', data = wcat).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(wcat['Waist']))
pred3

pred3_at = np.exp(pred3)
pred3_at

res3 = wcat.AT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

plt.scatter(wcat.Waist, np.log(wcat['AT']))
plt.plot(wcat.Waist, pred3, "r")
plt.legend(['Observed data', 'Predicted line'])


# model 4
# Quadratic Regression 

X = pd.DataFrame(wcat['Waist'])
Y = pd.DataFrame(wcat['AT'])

model4 = smf.ols('np.log(AT) ~ Waist + I(Waist * Waist)', data = wcat).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(wcat))
pred4

pred4_at = np.exp(pred4)
pred4_at

res4 = wcat.AT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4

plt.scatter(X['Waist'], np.log(Y['AT']))
plt.plot(X['Waist'], pred4, color = 'red')
plt.plot(X['Waist'], pred3, color = 'green')
plt.legend(['Transformation data', 'Polynomial Regression line', 'Linear Regression line'])


data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

# final model 
from sklearn.model_selection import train_test_split
train, test = train_test_split(wcat, test_size = 0.2, random_state = 0)


plt.scatter(train['Waist'], np.log(train['AT']))

plt.scatter(test['Waist'], np.log(test['AT']))

final_model = smf.ols('np.log(AT) ~ Waist + I(Waist * Waist)', data = train).fit() #⚠️
final_model.summary()


# prediction on test
pred_test = final_model.predict(test)
pred_test

pred_test_at = np.exp(pred_test)
pred_test_at

# model evaluation
res_test = test.AT - pred_test_at
res_test_sqr = res_test * res_test
mse_test = np.mean(res_test_sqr)
rmse_test= np.sqrt(mse_test)
rmse_test


# prediction on train
pred_train = final_model.predict(pd.DataFrame(train))
pred_train

pred_train_at = np.exp(pred_train)
pred_train_at

res_train = train.AT - pred_train_at
res_train_sqr = res_train * res_train
mse_train = np.mean(res_train_sqr)
rmse_train = np.sqrt(mse_train)
rmse_train

### more or less close to remse_test and rmse_train then it is Right Fit ###


# combine the whole data set (training & test) cause it is a right fit 
# build the model 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression
import pickle
poly_model = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
'''
Let me make a pipeline of polynomial features of degree 2.
Whenever you have inputs, it will take X and X Square. You want
to run a linear regression model.
'''
poly_model.fit(wcat[['Waist']], wcat[['AT']]) #fit on (X,Y) = complete data set

pickle.dump(poly_model, open('poly_model.pkl', 'wb'))


## testing on new data
# load the saved pipelines
impute = joblib.load('imputation')
winsor = joblib.load('winzor')

poly_model = pickle.load(open('poly_model.pkl', 'rb')) # just to check whether ur model is working well or not
poly_model

wcat_test = pd.read_csv(r"C:\Users\hruth\Desktop\self pace learning\my work\simple linear regression\5.c.Simple Linear Regression\wc-at_test.csv")

clean1 = pd.DataFrame(impute.transform(wcat_test), columns = wcat_test.select_dtypes(exclude = ['object']).columns)
clean2 = pd.DataFrame(winsor.transform(clean1), columns = clean1.columns)

prediction = pd.DataFrame(poly_model.predict(clean2), columns = ['Pred_AT'])

final = pd.concat([prediction, wcat_test], axis = 1)
final



























































