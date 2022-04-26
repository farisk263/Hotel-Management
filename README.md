# Classification on Hotel Booking Cancellation on Hotel Booking Demand Dataset using Machine Learning model

![Portugal & Hotel](https://videohive.img.customer.envatousercontent.com/files/9431b98a-d191-4101-ab26-4e7aac1c3e0e/inline_image_preview.jpg?auto=compress%2Cformat&fit=crop&crop=top&max-h=8000&max-w=590&s=0c359dd8c1702b5b8b8a9060e31cec1a)
## Introduction

A hotel dataset is given for this exercise. After a thorough research, the details of the dataset has been found. The dataset is filled with hotel demand data which is acquired from [this paper](https://www.sciencedirect.com/science/article/pii/S2352340918315191). There is two types of hotels from the dataset, one which is _H1_, a resort hotel and _H2_, a city hotel. Both hotels are located in Portugal, based on this paper. The data contains bookings due to arrive between the 1st of July 2015 and 31st of August 2017.

## Problem Statement
Hotels usually do not know the guests' booking or cancellation pattern & insights on the guests' inclination. It would be better if hotels can predict if a guest will come using a machine learning model.

## Descriptive Analytics

- ### Data Information
  - It has about 31 features & 1 target variable that will be used later for model training to predict either the guest would cancel the booking or not. 
  - The dataset consists of 20 numerical data & 12 categorical data, including the target variable.
  - The target variable would be `is_canceled` column which contain integer value of 1 and 0. 1 indicates guests will cancel the booking while 0 is not.
    
- ### EDA
  - **_How many repeated guests decided to cancel booking ?_**
 
    ![booking repeated_guests](https://user-images.githubusercontent.com/63250608/165352175-e00135e9-3f53-4d1a-a80a-3df7b796ba3b.png)
    
    ```
    Percentage of repeated guests who:- 
    Cancelled: 14.6471%
    Not Cancelled: 85.3529%
    ```
    
    We can pretty much understand that, mostly of the repeated guests tend to not cancel their bookings. About 85% of the repeated guests decided to proceed with their bookings and stayed at the respective hotels. Only ~15% of them cancelled the bookings because of any other unplanned events.
    
  - **_Identified how much guests paid for a night & how the price fluctuated over the year._**

    ```
    Average cost per person for each hotel per night for at average of all room types:- 
    Resort Hotel: 47.49 €
    City Hotel: 59.27 €
    ```
    
    Above are the average room costs per night regardless of the meal & room types. It only covers the actual guests who really come & stay at the hotels. It does not include guests who cancelled their booking. Price is in EUR as the hotels are operated in Portugal.
    
    ![average_room_price](https://user-images.githubusercontent.com/63250608/165353847-410128ef-0d94-45c1-834f-50306f342b86.png)
    
    The graph explained that during summer season (June - September), the price per night at Resort Hotel spiked as people tend to go to beaches. The demand spiked & triggered the price to jack up even more during that time. Moreover, the price at City Hotel peaked in May & September. The vertical line shown the max price for the respective hotels in a year.
    
  - **_Finding the difference of average total nights stayed between guests who has family & no family._**
  
    ![night_stay_family](https://user-images.githubusercontent.com/63250608/165354592-a6326459-709b-4539-a20b-046c9f3b6e49.png)
    
    Based on the graph above, we can understand that guests which came as family, tend to go for longer stay compared to individual / couple. This is totally understandable as family tends to go for longer vacation compared to individual, who some of them potentially just went for short business trip. We also can see that guests favored to stay longer at resort hotel than city hotel.
    
  - **_How long people stay at the hotels ?_**

    ![length_stay](https://user-images.githubusercontent.com/63250608/165354819-fed2d86a-dcb1-42f7-baf8-427e6e9a8705.png)
    
    Based on the graph above, both resort & city hotel, most of the guests stayed from 1 - 4 nights. However, 7 nights stay can be well liked by the resort hotel's guests as well. Minority of resort hotel's guests also liked to stay in until 14 nights stays.
    
  - **_Possibility of bookings got canceled due to high lead time_**

    ![lead_time_over_cancellation](https://user-images.githubusercontent.com/63250608/165355236-ef378083-b120-4bc0-85e6-697d81b0ea57.png)

    It shows that,number of people cancelling their booking is higher than not when the lead time is higher than 50 days (cancellation rate is higher at this point).
    
  - **_Find out the busiest month for hotels_**

    ![booking_trend](https://user-images.githubusercontent.com/63250608/165355548-06693be7-ee0b-4be2-80cf-c12d89801fa9.png)

    From graph above, both hotels are pretty much occupied in summer (June - September) due to holiday season. Bookings are pretty much less during winter season (December - March).
    
  - **_Find out the daily price relationship with length of stay_**
    
    ![length_stay_over_price](https://user-images.githubusercontent.com/63250608/165355842-bf26de5b-1b74-4a2b-b9cf-3f758691a130.png)

    Average daily price for city hotel spiked at 24 days length of stay. Most of the time, resort hotel daily room price can be considered much cheaper than city hotel daily rate. This may possible due to high land value in the city & high land rental cost which cause the daily room rate to be much expensive. Higher length of stay for resort hotel means cheaper price while city hotel's does not have the same effect.
    

## Model Development
  - Multiple machine learning classification models were tested against the dataset, eg: Random Forest, Logistic regression, XGBoost & Decision Tree


  - Evaluation

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

