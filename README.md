# Classification on Hotel Booking Cancellation on Hotel Booking Demand Dataset using Machine Learning model

![Portugal & Hotel](https://videohive.img.customer.envatousercontent.com/files/9431b98a-d191-4101-ab26-4e7aac1c3e0e/inline_image_preview.jpg?auto=compress%2Cformat&fit=crop&crop=top&max-h=8000&max-w=590&s=0c359dd8c1702b5b8b8a9060e31cec1a)
## **Introduction**

A hotel dataset is given for this exercise. After a thorough research, the details of the dataset has been found. The dataset is filled with hotel demand data which is acquired from [this paper](https://www.sciencedirect.com/science/article/pii/S2352340918315191). There is two types of hotels from the dataset, one which is _H1_, a resort hotel and _H2_, a city hotel. Both hotels are located in Portugal, based on this paper. The data contains bookings due to arrive between the 1st of July 2015 and 31st of August 2017.

## **Problem Statement**
Hotels usually do not know the guests' booking or cancellation pattern & insights on the guests' inclination. It would be better if hotels can predict if a guest will come using a machine learning model.

## **The Journey**

- **Data Information**
    - It has about 31 features & 1 target variable that will be used later for model training to predict either the guest would cancel the booking or not. 
    - The dataset consists of 20 numerical data & 12 categorical data, including the target variable.
    - The target variable would be `is_canceled` column which contain integer value of 1 and 0. 1 indicates guests will cancel the booking while 0 is not.
    
- **Architecture**
  - Multiple machine learning classification models were tested against the dataset, eg: Random Forest, Logistic regression, XGBoost & Decision Tree

- **Evaluation**

    - Train Normal

    ![Train Normal](https://user-images.githubusercontent.com/63250608/164382689-5b847d93-586f-4ab0-9a1d-e97316847027.png)
    
    - Test Normal
    
    ![Test Normal](https://user-images.githubusercontent.com/63250608/164382754-881c1f42-1d45-42ab-a657-d33ba200db4e.png)


    - Train K-Fold
    
    ![Train K5](https://user-images.githubusercontent.com/63250608/164382844-171bf913-476b-444e-b4d2-c4403f17ea00.png)

    
    - Test K-Fold
    
    ![Test K5](https://user-images.githubusercontent.com/63250608/164382919-1d60cca8-eaec-4b78-b0e5-78d9ac21180e.png)

    
    - Average F1 Score 
    
    ![Average F1 all fold](https://user-images.githubusercontent.com/63250608/164382977-a3b62b84-d086-490a-a47e-c5259480df28.png)
    
    
## **Conclusion**
 
  - K-Folding improve the overall score but longer processing time.
  - The effect of lack of equal representation for all the classes cannot be improve greatly using synthetic data. 

    
## **Future Improvement**
  - Pick a more robust and comprehensive data as the model cannot be improved using such an imbalance data.
  - Needs more representation of other classes.
  - Using arbiter to test more variables and hyperparameter.



## Getting Started 

- Clone a copy of the repository. 

```

git clone https://github.com/farisk263/Red-Wine-Quality-Check.git

```
- Using an IDE and open the RedWineQualityCheck.java




## Built on

* [Maven](https://maven.apache.org/) - Dependency Management
* [DL4J](https://deeplearning4j.org/) - Deep Learning Library


## FAQ 

If you are having problems on installing Maven dependencies. Try to reload it by right-clicking the POM.xml

