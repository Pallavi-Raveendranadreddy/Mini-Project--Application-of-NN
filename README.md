# Mini-Project--Application-of-NN

## Project Title:
Implementation of MLP classifier for Rasins
## Project Description :
A raisin is a dried grape. Raisins are produced in many regions of the world and may be eaten raw or used in cooking, baking, and brewing. In the United Kingdom, Ireland, New Zealand, and Australia, the word raisin is reserved for the dark-colored dried large grape,[1] with sultana being a golden-colored dried grape, and currant being a dried small Black Corinth seedless grape.In this experiment we contain dataset called rasins_dataset.In that dataset we will contain information of rasins like
area,major axis length,minor axis length,eccentricity,convex area,extent,perimeter,class.WE will find mlp classifier for rasins.
## Algorithm:
1.Import the necessary libraries of python.

2.After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3.Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4.Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%. Normalize our dataset.

5.In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6.Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7.In order to get the predicted values we call the predict() function on the testing data set.

8.Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.
## Program:
Develped by:Chevula.Nagadurga

Regno:212221230014

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("Raisin_Dataset.csv")
data.head()
name=["Area","MajorAxisLength","MinorAxisLength","Eccentricity","ConvexArea","Extent","Perimeter","Class"]
x=data.iloc[:,0:4]
y=data.select_dtypes(include=[object])
x.head()
y.head()
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data['Class']=label_encoder.fit_transform(data['Class'])
data['Class'].unique()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
mlp.fit(x_train,y_train.values.ravel())
predictions=mlp.predict(x_test)
print(predictions)
```
## Output:
![output](https://github.com/Nagadurg/Mini-Project--Application-of-NN/blob/93f87d73eeb2c244a2d5371836bf50680dbe72d6/o2.png)

![output](https://github.com/Nagadurg/Mini-Project--Application-of-NN/blob/93f87d73eeb2c244a2d5371836bf50680dbe72d6/o3.png)

![output](https://github.com/Nagadurg/Mini-Project--Application-of-NN/blob/93f87d73eeb2c244a2d5371836bf50680dbe72d6/o4.png)

![output](https://github.com/Nagadurg/Mini-Project--Application-of-NN/blob/93f87d73eeb2c244a2d5371836bf50680dbe72d6/o5.png)

![output](https://github.com/Nagadurg/Mini-Project--Application-of-NN/blob/93f87d73eeb2c244a2d5371836bf50680dbe72d6/o6.png)

## Advantage :

Raisins amazing health benefits includes treating anemia, preventing cancer, promoting proper digestion, combating hair loss, treating skin diseases, treating joint pains, regulating body pH level, relieving fever, support eye health, boosting energy level, supporting dental health, and curing insomnia.
## Result:
Thus Implementation-of-MLP-with-Backpropagation problem is executed successfully.


