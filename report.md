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
- Splitting data more accurately by including pca results  
- Adding idaweb data (e.g. pressure, clouds) as an input 
- Evaluate the performance of the models obtained

