# Problem Introduction
L'objectif de ce projet est de détecter la présence de la couche et de prévoir à court terme (prévision pour les 6 prochaines heures) sa disparition ou son apparition. Le stratus est une accumulation de nuages présente dans le canton de Vaud. La particularité de cette accumulation de nuages est qu'elle est très basse, donc si nous pouvions nous déplacer sur une montagne suffisamment haute, nous pourrions voir une accumulation de nuages recouvrant la plaine du canton de Vaud. Heureusement pour nous, nous disposons d'une caméra, grâce aux images de météo suisse, qui nous permet de nous placer juste au-dessus du startus et d'avoir une bonne vue de la plaine vaudoise. Cette caméra est située à la Dôle et sera utilisée dans le cadre de ce projet. Afin de mieux comprendre les images dont nous disposons et le phénomène météorologique que nous essayons d'étudier, nous allons examiner de quoi nous parlons. Voici quelques exemples d'images disponibles. [TODO] METTRE DES PHOTOS DE LA DÔLE.


Maintenant que nous avons une vue d'ensemble du problème, nous pouvons comprendre plus en détail quelles sont les caractéristiques de ce phénomène météorologique.
Voici quelques recherches effectuées: 

### Différence entre le brouillard et le stratus :
- **Distance du sol** :  
- **Visibilité** :
    - Brouillard :  < 1 km  
    - Stratus : > 1 km  
(Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/portrait/meteosuisse-blog/fr/2025/02/grisailles-frequentes-semestre-hiver.html))

### Facteurs qui affectent la présence du brouillard ou du stratus :
(Source : [Météonews](https://meteonews.ch/fr/News/N14196/Brouillard-et-stratus-%E2%80%93-Compagnons-de-la-saison-froide)
)
- **La direction du vent** :  
    - Si le vent vient du **Nord-Est** : c'est la bise (transport de l'air continental → froid et lourd). Comme l'air est froid et lourdes, le brouillard monte et crée ainsi du stratus.  
    - **Si la bise est plus forte** on a un stratus avec une limite supérieure plus haute.
    - **Bise faible** : limite supérieure < 1000 m.
    - **Bise moyenne** : limite supérieure ≤ 1500 m.
    - **Bise forte** : limite supérieure > 2000 m.  


### Comment le stratus disparaît-il ?
- **Grâce au rayonnement solaire** : Le soleil chauffe la couche froide, ce qui rend la dissipation plus difficile en hiver.
- **Le vent** : Si le vent amène de l'air chaud (vent de sud-ouest à ouest) ou de l'air sec (foehn).
- **Arrivée d'une perturbation** : Changement de pression ou de température.
- **Modification de la pression atmosphérique**.
- Plus la limite supérieure est élevée, plus les chances de dissolution sont faibles, mais d'autres facteurs jouent également un rôle.  
(Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/portrait/meteosuisse-blog/fr/2024/10/limite-superieure-brouillard.html))

### Formation du stratus :  
Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/meteo/meteo-et-climat-de-a-a-z/brouillard/le-plateau-une-region-a-brouillard.html#:~:text=,remplie%20sur%20le%20Plateau%20suisse)  
La bise est souvent présente lors de la formation du stratus.  
Source : [Agrometeo](https://api.agrometeo.ch/storage/uploads/Web_Wetterlagen_FR_low.pdf)

- Dans une situation anticyclonique stable, il y a une **inversion thermique**.  
- Dans le cas du stratus, on a une inversion thermique (l'air est plus chaud en hauteur qu'au sol). Cette inversion est créée par des anticyclones. L'air froid emprisonné près du sol est souvent humide, surtout après des nuits claires où le refroidissement radiatif est important. Lorsque l'humidité atteint le point de saturation, elle se condense et forme des nuages bas appelés stratus.  
  - **Inversion thermique** : Se forme avec une situation anticyclonique stable, qui prévoit des pressions atmosphériques élevées, poussant l'air froid en bas (source : [MétéoSuisse](https://www.meteosuisse.admin.ch/meteo/meteo-et-climat-de-a-a-z/brouillard/le-plateau-une-region-a-brouillard.html#:~:text=,remplie%20sur%20le%20Plateau%20suisse)) → Données très importantes, il faut regarder la pression.
- Faible ensoleillement ou soleil bas.
- Vent faible dans les couches basses de l’atmosphère (à l’exception de la bise) : condition satisfaite en situation de haute pression.
- **Topographie** : L’air froid et humide doit pouvoir s’accumuler dans un bassin (ex. la région suisse entre les Alpes et le Jura) → ce qui crée une inversion thermique.
- Ciel clair → satisfait en conditions de haute pression.
- Humidité élevée : Comme l’air froid peut contenir moins d’humidité que l’air chaud, la vapeur d’eau finit par se condenser et former du brouillard ou du stratus.

### La saison influence la présence du stratus :
Le stratus est plus présent en **hiver** et **automne**, et la vitesse à laquelle il peut disparaître varie.  
*À voir si diviser les études en saisons pourrait avoir des avantages.*

Comme on le voit, il existe donc plusieurs facteurs météorologiques qui influencent l'apparition et la disparition du stratus. Actuellement, Météo Suisse utilise certaines de ces données pour prédire ce phénomène météorologique. Malheureusement, dans certains cas, ces données ne sont pas suffisantes ou nécessitent une intervention humaine pour garantir l'exactitude des prévisions et les réaliser de manière autonome.

Dans ce projet, je vais donc essayer de créer un modèle qui, au lieu d'utiliser uniquement les données météorologiques, utilise également les images de Dôle afin d'obtenir le plus d'informations possible. 
Pour réaliser ce projet, je dispose de deux années d'images et de deux années de données météorologiques Inca avec un timestamp de 10 minutes, plus des données de la plateforme idaweb, afin d'obtenir des informations supplémentaires qui ne sont pas disponibles dans les données Inca.

Dans notre cas, comment pouvons-nous déterminer s'il y a un stratus à un moment donné ? Pour résumer, ce phénomène est un phénomène qui obscurcit la plaine vaudoise, ce qui veut dire que dans la plaine nous n'aurons pas de soleil, contrairement à la Dôle où il y aura du soleil. Donc, pour savoir si on est en présence d'un stratus à un moment donné, il suffit de mesurer la puissance d'irradiation donnée par le soleil à la Dôle et en plaine. Les deux données d'irradiation, ainsi que toutes les autres données météorologiques, sont disponibles grâce aux stations météorologiques de la dôle et de nyon (en plaine). Si la différence de rayonnement entre dôle et nyon est importante, cela signifie que nous avons un startus.


On pourrait alors mesurer la couverture nuageuse des deux stations, mais pour l'instant on ne dispose que d'un pourcentage qui indique à quel point le ciel est couvert et qui est beaucoup moins fiable (voici les résultats). La prévision des stratus nocturnes est donc un cas isolé que nous laissons de côté pour l'instant.
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

# 2025-06-05

## Tasks

1. I found a model size that gives acceptable results. In the output of the cnn I have a size of 16*16*32.
2. I added the second view of the dole camera.
3. Testing different models from which I selected the following.
    1. "model_14" --> 1 view, prediction for the next 10 minutes
    2. "model_16" --> 2 view, prediction for the next 10 minutes
    3. "model_15" --> 2 view, prediction for the next 10 minutes, with a larger model size
    4. "model_3" and "model_4" --> 2 view, prediction for the next 10 minutes with a final MLP bigger 
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
        
As we can see there are improvements in rmse but not in relative error so we can conclude that it shows no signs of strong improvement. In model 4 I increased the number of neurons in the mlp even more but gave no significant changes
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


4. Image normalization

I then focused on the fact that the images were not being normalized for the moment. So I decided to normalize them to allow the model to have the same range of values for the input values. I tried various combinations of models structure with 1 camera view all showing decreasing loss without overfitting but no one allowed me to get better results than model_14.

## To do for next time:

1. Add the time series of images and weather data to the model
2. Find right model size
3. Add statistics to better evaluate the model
4. Add images to daily curves 