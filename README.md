# Crime-Rate-Analysis
## Henry Harvin Internship Project


## INTRODUCTION:

Crime is one of the biggest and dominating problem in our society and its prevention is an important. task. Daily there are huge numbers of crimes committed frequently. Perhaps it is increasing and spreading at a fast and vast rate. Crimes happen from small village, town to big cities. Crimes are of different type – robbery, murder, rape, assault, battery, false imprisonment, kidnapping, homicide. Since crimes are increasing there is a need to solve the cases in a much faster way. The crime activities have been increased at a faster rate and it is the responsibility of police department to control and reduce the crime activities. Crime prediction and criminal identification are the major problems to the police department as there are tremendous amount of crime data that exist. There is a need of technology through which the case solving could be faster. This require keeping track of all the crimes and maintaining a database for same which may be used for future reference. The current problem faced are maintaining of proper dataset of crime and analysing this data to help in predicting and solving crimes in future.

## AIM:
The aim of this project is to make crime prediction using the features present in the dataset. The dataset is extracted from the official sites. With the help of machine learning algorithm, using python as core we can predict the type of crime which will occur in a particular area. The objective would be to train a model for prediction. The training would be done using the training data set which will be validated using the test dataset. Building the model will be done using better algorithm depending upon the accuracy. The K-Nearest Neighbour (KNN) classification and other algorithm will be used for crime prediction. Visualization of dataset is done to analyse the crimes which may have occurred in the country. This work helps the law enforcement agencies to predict and detect crimes with improved accuracy and thus reduces the crime rate

## CONCEPTS USED:
1) Predictive modelling: Predictive analytics encompasses a variety of statistical techniques from data mining, predictive modelling, and machine learning that analyse current and historical facts to make predictions about future or otherwise unknown events.Predictive modelling is the way of building a model that is capable of making predictions. The process includes a machine learning algorithm that learns certain properties from a training dataset in order to make those predictions.
Predictive modelling can be divided further into two areas: Regression and Classification. 
* Regression - When the output is a numerical variable like income, the model used is of regression i.e., for continuous data. The aim of this algorithm is to find a    mapping function to map the input independent variable(x) with the output dependent variable(y). E.g. To find the price of a house, its dependence on features like area in sqrft, no. of bedrooms, etc. will be considered and a function will be created. For any input, we will be able to calculate the numerical output using the calculated function. Predicted output is compared with observed output for accuracy.

* Classification - When the output is a categorical variable like yes/no i.e. discrete data, we use classification. The objective of this model is to predict the correct label for newly presented input data. We evaluate the input data to draw a relation between dependant and independent variables. E.g. to find out whether a fruit is poisonous or not, its dependence on colour, smell, origin, rigidity and taste will be considered. Depending upon the inferences drawn, the input is classified as poisonous or not. The output is a categorical variable.
Classification tasks can be divided into two parts, Supervised and unsupervised learning. In supervised learning, the class labels in the dataset, which is used to build the classification model, are known. In a supervised learning problem, we would know which training dataset has the particular output which will be used to train so that prediction can be made for unseen data.



2) Data Collection:
The data set can be collected from various sources such as a file, database, sensor and many other such sources but the collected data cannot be used directly for performing the analysis process as there might be a lot of missing data, extremely large values, unorganized text data or noisy data. 


3) Data Pre-processing:
Data pre-processing is a process of cleaning the raw data i.e. the data is collected in the real world and is converted to a clean data set. In other words, whenever the data is gathered from different sources it is collected in a raw format and this data isn’t feasible for the analysis.
Therefore, certain steps are executed to convert the data into a small clean data set, this part of the process is called as data pre-processing.
Data Preprocessing can be done in following ways:
* **Conversion of data**: As we know that Machine Learning models can only handle numeric features, hence categorical and ordinal data must be somehow converted into numeric features.
* **Ignoring the missing values**: Whenever we encounter missing data in the data set then we can remove the row or column of data depending on our need. This method is known to be efficient but it shouldn’t be performed if there are a lot of missing values in the dataset.
* **Filling the missing values**: Whenever we encounter missing data in the data set then we can fill the missing data manually, most commonly the mean, median or highest frequency value is used.
* **Feature Generation**: If we have some missing data then we can predict what data shall be present at the empty position by using the existing data.
* **Outlier’s detection**: There are some error data that might be present in our data set that deviates drastically from other observations in a data set. [Example: human weight = 800 Kg; due to mistyping of extra 0]


4) Random Sampling:
For training a model we initially split the model into 2 three sections which are ‘Training data’ and ‘Testing data’.
We train the classifier using ‘training data set’ and then test the performance of your classifier on unseen ‘test data set’. An important point to note is that during training the classifier only the training set is available. The test data set must not be used during training the classifier. The test set will only be available during testing the classifier.


5) Model Selection:
Based on the defined goal(s) (supervised or unsupervised) we have to select one of or combinations of modelling techniques. Such as 
* KNN Classification 
* Logistic Regression 
* Decision Trees 
* Random Forest 
* Support Vector Machine (SVM) 



6) Confusion matrix:
Once the model is trained, we can use the same trained model to predict using the testing data i.e. the unseen data. Once this is done, we can develop a confusion matrix, this tells us how well our model is trained. A confusion matrix has 4 parameters, which are ‘True positives’, ‘True Negatives’, ‘False Positives’ and ‘False Negative’. We prefer that we get more values in the True negatives and true positives to get a more accurate model. The size of the Confusion matrix completely depends upon the number of classes.

* True positives: These are cases in which we predicted TRUE and our predicted output is correct.
* True negatives: We predicted FALSE and our predicted output is correct.
* False positives: We predicted TRUE, but the actual predicted output is FALSE.
* False negatives: We predicted FALSE, but the actual predicted output is TRUE.
* We can also find out the accuracy of the model using the confusion matrix.
 Accuracy = (True Positives +True Negatives) / (Total number of classes)


7) Model Evaluation:
Model Evaluation is an integral part of the model development process. It helps to find the best model that represents our data and how well the chosen model will work in the future.


## IMPLEMENTATION OF THE PROJECT:
The dataset used in this project is given by Henry Harvin. The implementation of this project is divided into following steps –
1.	Data collection:
 Crime dataset from HH is used in CSV format.
2.	Data Pre-processing:
10 lac entries are present in the dataset. The null values are removed using df = df. dropna () where df is the data frame. The categorical attributes (Primary Type, Location) are converted into numeric using Binary and BaseN Encoders. The date attribute is splitted into new attributes like month, day and hour which can be used as feature for the model. 
3.	 Feature selection
Features selection is done which can be used to build the model. The attributes used for feature selection are Domestic, Location, Primary Type, Hour, day and month.
4.	Building and Training Model:
 After feature selection location and month attribute are used for training. The dataset is divided into pair of xtrain ,ytrain and xtest, y test. The algorithms model is imported form skleran. Building model is done using model. Fit (xtrain, ytrain). 
5.	Prediction:
 After the model is build using the above process, prediction is done using model.predict(xtest). The accuracy, fscore, precision is calculated.

6.	Visualization:
 Using matplotlib library from sklearn. Analysis of the crime dataset is done by plotting various graphs. 



## ACCURACY OBTAINED:
* Accuracy of Decision tree **0.7621539314848439**
* Accuracy of Random Forest **0.8161618117219835**
* Accuracy of Logistic Regression **0.8109501807849888**
* Accuracy of KNN **0.816360649046889**
* Accuracy of SVM **0.8109214016984894**


## CONCLUSION:
With the help of machine learning technology, it has become easy to find out relation and patterns among various data. The work in this project mainly revolves around predicting the type of crime which may happen if we know the location of where it has occurred. Using the concept of machine learning we have built a model using training data set that have undergone data cleaning and data transformation. The model predicts the type of crime with accuracy of 0.816. Data visualization helps in analysis of data set. The graphs include bar, pie, line and scatter graphs each having its own characteristics. We generated many graphs and found interesting statistics that helped in understanding crimes datasets that can help in capturing the factors that can help in keeping society safe.

