# 2025-05-27

## Tasks

### 1.  Data analysis
- Analysis of days on which the stratus is present 
- Isolation of a few days when the stratus is present through median analysis (calculation of the delta between nyon or dôle irradiation and pca study)
- ### Analysis : 

![Daily difference between Dôle and Nyon](analysis/diff/daily_difference_2024-10.png)

We can see that there are peaks. High peaks indicate days where there is the presence of a status.

We can see what these days correspond to in more detail 
#### Stratus days
![2024-10-26](analysis/2024-10-26.png)
![2024-10-30](analysis/2024-10-30.png)


#### Half stratus condition

![2024-10-25](analysis/2024-10-25.png)
![2024-10-20](analysis/2024-10-20.png)


However, the daily difference still hides some information. It is difficult to establish a delta threshold at which the layer disappears or appears because this threshold can change between months. With the help of pca we can extract information that enables us to better recognise days with or without a stratus.

#### Example of PCA 2024-10


![2024-10](analysis/pca/pca_gre000z0_nyon_dole_2024-10.png)

In this analysis made on the month of October, we can mainly see three groups that allow us to isolate the days with the stratus. In fact, the days with the stratus are in the negative y-axis. The more negative the value is, the more the layer tends to remain all day. The closer the value is to y = 0 ‘the more the stratus will tend to disappear. (See layer days described above). Another case where the layer tends to disappear is when the day is in negative y but is very distant from x = 0
When the y-axis is positive, we have the reverse situation at the layer (information that we can also find on the daily difference graph).
As far as the x-axis is concerned, the more days are on the negative side, the worse the weather will be. 

We can prove this by looking at the pca of 2023-06 where we can take as an example

2023-06-30 --> bad weather 

2023-06-14 --> good weather


![2023-06](analysis/pca/pca_gre000z0_nyon_dole_2023-06.png)

![2023-06-30](analysis/2023-06-30.png)
![2023-06-14](analysis/2023-06-14.png)

As expected, we can also see that startus days are not very present in summer due to the low occurrence of days with negative y and x close to 0
### 2.  Model size Increased
- Model size increased since last time but not yet the target size   
### 3.  Data preparation 
- Data normalization (min-max)
- Pseudo-random splitting of data, being sure to have a good percentage of days with and without layer in train and test
### 4. Code written and tested for training 
### 5. Code written and tested for inference
### 6. Some statistics done(mae, mse, accurancy, relative error for each point)
### 7. Setup on chacha done and first train to test launched. Not good results probably due to the size of the neural network which is too small 

The loss is: 

![loss](models/model_1/loss.png)

## For the next time 
- Change model size
- Splitting data more accurately 
- Adding idaweb data (e.g. pressure, clouds) as an input 
- Evaluate the performance of the models obtained

# 2025-05-05

## Tasks

1. I found a model size that gives acceptable results. In the output of the cnn I left a size of 16*16*32.
2. I added the second view of the dole camera.
3. Testing different models from which I selected the following.
    1. "model_14" --> 1 view, prediction for the next 10 minutes
    2. "model_16" --> 2 view, prediction for the next 10 minutes
    3. "model_15" --> 2 view, prediction for the next 10 minutes, with a larger model size
    4. "model_3"  --> 2 view, prediction for the next 10 minutes with a final MLP bigger 
    5. "model_5" --> 1 view, prediction for the next 60 minutes with the same model stucture of model_14 
### Results analysis

#### Model Performance Metrics model_14
- **Loss**
![str_1](models/model_14/loss.png)
- **Root Mean Squared Error (RMSE):**
    - Nyon: **86.55**
    - Dôle: **79.28**

- **Mean Relative Error:**
    - Nyon: **0.49**
    - Dôle: **0.52**

- **Stratus Days:**
    - RMSE
        - Nyon: **60.17**
        - Dôle: **61.28**
    - Relative Error
        - Nyon: **0.57**
        - Dôle: **0.17**

- **Non-Stratus Days:**
    - RMSE
        - Nyon: **70.44**
        - Dôle: **67.32**
- **Observations of clear stratus days:**

![str_1](models/model_14/metrics/2023-01/day_curve_2023-01-26.png)
![str_2](models/model_14/metrics/2023-01/day_curve_2023-01-24.png)
![str_4](models/model_14/metrics/2023-12/day_curve_2023-12-17.png)
![str_4](models/model_14/metrics/2024-11/day_curve_2024-11-04.png)

- **Observations of half stratus days:**


![str_3](models/model_14/metrics/2023-09/day_curve_2023-09-02.png)
![str_2](models/model_14/metrics/2024-10/day_curve_2024-10-25.png)
![str_2](models/model_14/metrics/2024-10/day_curve_2024-10-28.png)

**Next improvement**
Problems with trend reversals :
![str_2](models/model_14/metrics/2024-11/day_curve_2024-11-09.png)
![str_2](models/model_14/metrics/2024-12/day_curve_2024-12-20.png)
![str_2](models/model_14/metrics/2024-12/day_curve_2024-12-09.png)
![str_4](models/model_14/metrics/2023-12/day_curve_2023-12-22.png)
#### Model Performance Metrics model_15

In model 15 I wanted to increase the cnn channel size from 32 to 64 and increased the mlp size what gave worse results so we will not analyze this model

#### Model Performance Metrics model_16
In front of these performances I decided to integrate a second view of the dole camera not touching to the structure of the model 14. 
Obtaining the following perfromances:
- **Loss**
![str_1](models/model_16/loss.png)
- **Mean Absolute Error (MAE):**
    - Nyon: **53.20**
    - Dôle: **62.51**

- **Root Mean Squared Error (RMSE):**
    - Nyon: **74.40**
    - Dôle: **85.32**

- **Mean Relative Error:**
    - Nyon: **0.44**
    - Dôle: **0.44**

- **Stratus Days:**
    - RMSE
        - Nyon: **64.55**
        - Dôle: **98.31**
    - Relative Error
        - Nyon: **0.60**
        - Dôle: **0.24**

- **Non-Stratus Days:**
    - RMSE
        - Nyon: **64.48**
        - Dôle: **71.66**
    - Relative Error
        - Nyon: **0.49**
        - Dôle: **0.48**
- **Observations of clear stratus days:**

According with statistics, the model performs less well

**Examples of obvious stratus that works not well**
![str_2](models/model_16/metrics/2023-01/day_curve_2023-01-24.png)
![str_2](models/model_16/metrics/2023-01/day_curve_2023-01-27.png)


**Examples of days on which it works well**
![str_2](models/model_16/metrics/2024-01/day_curve_2024-01-14.png)
![str_2](models/model_16/metrics/2024-01/day_curve_2024-01-13.png)

#### Model Performance Metrics model_3
- **Loss**
![str_2](models/model_3/loss.png)
- **Mean Absolute Error (MAE):**
    - Nyon: **51.96**
    - Dôle: **62.27**

- **Root Mean Squared Error (RMSE):**
    - Nyon: **75.31**
    - Dôle: **84.12**

- **Mean Relative Error:**
    - Nyon: **0.44**
    - Dôle: **0.48**

- **Stratus Days:**
    - RMSE
        - Nyon: **55.53**
        - Dôle: **89.27**
    - Relative Error
        - Nyon: **0.63**
        - Dôle: **0.37**

- **Non-Stratus Days:**
    - RMSE
        - Nyon: **63.55**
        - Dôle: **71.67**
    - Relative Error
        - Nyon: **0.45**
        - Dôle: **0.48**
        
As we can see there are improvements in rmse but not in relative error so we can conclude that it shows no signs of strong improvement
#### Model Performance Metrics model_5

The best results were obtained from model 14, I decided to keep the same model structure to try and make a startus prediction for the next hour

The results are the following : 

- **Loss**
- **Loss**
![str_2](models/model_5/loss.png)

- **Mean Absolute Error (MAE):**
    - Nyon: **84.30**
    - Dôle: **92.12**

- **Root Mean Squared Error (RMSE):**
    - Nyon: **116.06**
    - Dôle: **123.38**

- **Mean Relative Error:**
    - Nyon: **0.67**
    - Dôle: **0.59**

- **Stratus Days:**
    - RMSE
        - Nyon: **92.48**
        - Dôle: **106.93**
    - Relative Error
        - Nyon: **0.66**
        - Dôle: **0.29**

- **Non-Stratus Days:**
    - RMSE
        - Nyon: **97.83**
        - Dôle: **101.04**


We can see that the rmse and relative error have increased. This increase has clearly impacted the performance


**Examples of obvious stratus that works well**
![str_2](models/model_5/metrics/2023-01/day_curve_2023-01-06.png)
![str_2](models/model_5/metrics/2024-11/day_curve_2024-11-04.png)



**Examples of days on which it works well**
![str_2](models/model_5/metrics/2024-10/day_curve_2024-10-28.png)
![str_2](models/model_5/metrics/2024-11/day_curve_2024-11-13.png)
![str_2](models/model_5/metrics/2024-03/day_curve_2024-03-27.png)


## To do for next time:

1. Add the time series of images and weather data to the model
2. Find right model size