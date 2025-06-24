# Introduction au problème
L'objectif de ce projet est de détecter la présence de la couche de stratus et de prévoir à court terme (prévision pour les 6 prochaines heures) sa disparition ou son apparition. Le stratus est une accumulation de nuages présente dans le canton de Vaud. Sa particularité est d’être très bas : si l’on se place sur une montagne suffisamment haute, on peut observer une mer de nuages recouvrant la plaine vaudoise. Heureusement, nous disposons d’une caméra, grâce aux images de MétéoSuisse, qui nous permet de nous placer juste au-dessus du stratus et d’avoir une bonne vue de la plaine vaudoise. Cette caméra est située à la Dôle et sera utilisée dans le cadre de ce projet.

Afin de mieux comprendre les images dont nous disposons et le phénomène météorologique étudié, nous allons préciser le contexte. Voici quelques exemples d’images disponibles :

![Non Stratus](analysis/non_startus_dole.jpeg)
![Stratus](analysis/startus_dole.jpeg)

Maintenant que nous avons une vue d’ensemble du problème, nous pouvons détailler les caractéristiques de ce phénomène météorologique.

## Différence entre le brouillard et le stratus

- **Distance du sol** :
- **Visibilité** :
    - Brouillard : < 1 km
    - Stratus : > 1 km  
(Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/portrait/meteosuisse-blog/fr/2025/02/grisailles-frequentes-semestre-hiver.html))

## Facteurs influençant la présence du brouillard ou du stratus

(Source : [Météonews](https://meteonews.ch/fr/News/N14196/Brouillard-et-stratus-%E2%80%93-Compagnons-de-la-saison-froide))

- **Direction du vent** :
    - Si le vent vient du **Nord-Est** : c’est la bise (transport d’air continental → froid et lourd). Comme l’air est froid et lourd, le brouillard monte et crée ainsi du stratus.
    - **Bise plus forte** : stratus avec une limite supérieure plus haute.
    - **Bise faible** : limite supérieure < 1000 m.
    - **Bise moyenne** : limite supérieure ≤ 1500 m.
    - **Bise forte** : limite supérieure > 2000 m.

## Comment le stratus disparaît-il ?

- **Grâce au rayonnement solaire** : Le soleil chauffe la couche froide, ce qui rend la dissipation plus difficile en hiver.
- **Le vent** : Si le vent amène de l’air chaud (vent de sud-ouest à ouest) ou de l’air sec (foehn).
- **Arrivée d’une perturbation** : Changement de pression ou de température.
- **Modification de la pression atmosphérique**.
- Plus la limite supérieure est élevée, plus les chances de dissolution sont faibles, mais d’autres facteurs jouent aussi un rôle.  
(Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/portrait/meteosuisse-blog/fr/2024/10/limite-superieure-brouillard.html))

## Formation du stratus

Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/meteo/meteo-et-climat-de-a-a-z/brouillard/le-plateau-une-region-a-brouillard.html#:~:text=,remplie%20sur%20le%20Plateau%20suisse)  
La bise est souvent présente lors de la formation du stratus.  
Source : [Agrometeo](https://api.agrometeo.ch/storage/uploads/Web_Wetterlagen_FR_low.pdf)

- En situation anticyclonique stable, il y a une **inversion thermique**.
- Dans le cas du stratus, on a une inversion thermique (l’air est plus chaud en hauteur qu’au sol), créée par les anticyclones. L’air froid emprisonné près du sol est souvent humide, surtout après des nuits claires où le refroidissement radiatif est important. Lorsque l’humidité atteint le point de saturation, elle se condense et forme des nuages bas appelés stratus.
    - **Inversion thermique** : Se forme avec une situation anticyclonique stable, qui prévoit des pressions atmosphériques élevées, poussant l’air froid en bas ([MétéoSuisse](https://www.meteosuisse.admin.ch/meteo/meteo-et-climat-de-a-a-z/brouillard/le-plateau-une-region-a-brouillard.html#:~:text=,remplie%20sur%20le%20Plateau%20suisse)).  
      → Données très importantes, il faut regarder la pression.
- Faible ensoleillement ou soleil bas.
- Vent faible dans les basses couches de l’atmosphère (sauf la bise) : condition satisfaite en situation de haute pression.
- **Topographie** : L’air froid et humide doit pouvoir s’accumuler dans un bassin (ex. la région suisse entre les Alpes et le Jura) → ce qui crée une inversion thermique.
- Ciel clair → satisfait en conditions de haute pression.
- Humidité élevée : Comme l’air froid peut contenir moins d’humidité que l’air chaud, la vapeur d’eau finit par se condenser et former du brouillard ou du stratus.

## Influence de la saison

Le stratus est plus présent en **hiver** et **automne**, et la vitesse à laquelle il peut disparaître varie.  
*À voir si diviser les études en saisons pourrait avoir des avantages.*

---

Comme on le voit, plusieurs facteurs météorologiques influencent l’apparition et la disparition du stratus.
On pourrait alors mesurer la couverture nuageuse des deux stations, mais pour l’instant on ne dispose que d’un pourcentage qui indique à quel point le ciel est couvert, ce qui est beaucoup moins fiable (voici les résultats). La prévision des stratus nocturnes est donc un cas isolé que nous laissons de côté pour l’instant.

# 2025-05-27

## Tâches

### 1. Analyse des données
- Analyse des jours où le stratus est présent
- Isolement de quelques jours où le stratus est présent via une analyse de la médiane (calcul du delta entre l’irradiation de Nyon ou Dôle et étude PCA)
- ### Analyse :

![Différence journalière entre Dôle et Nyon](analysis/diff/daily_difference_2024-10.png)

On observe des pics. Les pics élevés indiquent des jours où le stratus est présent.

On peut voir à quoi correspondent ces jours plus en détail :
#### Jours de stratus
![2024-10-26](analysis/2024-10-26.png)
![2024-10-30](analysis/2024-10-30.png)

#### Condition de demi-stratus

![2024-10-25](analysis/2024-10-25.png)
![2024-10-20](analysis/2024-10-20.png)

Cependant, la différence journalière masque encore certaines informations. Il est difficile d’établir un seuil de delta à partir duquel la couche disparaît ou apparaît, car ce seuil peut changer selon les mois. Avec l’aide de la PCA, on peut extraire des informations permettant de mieux reconnaître les jours avec ou sans stratus.

#### Exemple de PCA 2024-10

![2024-10](analysis/pca/pca_gre000z0_nyon_dole_2024-10.png)

Dans cette analyse faite sur le mois d’octobre, on distingue principalement trois groupes permettant d’isoler les jours avec stratus. En effet, les jours avec stratus sont sur l’axe y négatif. Plus la valeur est négative, plus la couche tend à rester toute la journée. Plus la valeur est proche de y = 0, plus le stratus tend à disparaître (voir les jours décrits ci-dessus). Un autre cas où la couche tend à disparaître est lorsque le jour est en y négatif mais très éloigné de x = 0.
Quand l’axe y est positif, on a la situation inverse du stratus (information que l’on retrouve aussi sur le graphe de la différence journalière).
Concernant l’axe x, plus les jours sont du côté négatif, plus la météo est mauvaise.

On peut le prouver en regardant la PCA de 2023-06 où l’on peut prendre comme exemple :

2023-06-30 --> mauvais temps

2023-06-14 --> beau temps

![2023-06](analysis/pca/pca_gre000z0_nyon_dole_2023-06.png)

![2023-06-30](analysis/2023-06-30.png)
![2023-06-14](analysis/2023-06-14.png)

Comme attendu, on constate aussi que les jours de stratus sont peu présents en été à cause de la faible occurrence de jours avec y négatif et x proche de 0.

Néanmoins, même une telle étude pca peut faire peNéanmoins, même une étude de pca de ce type et l'étude de la médiane peuvent faire perdre beaucoup d'informations. Pour l'instant, j'utilise donc la médiane pour filtrer les jours de stratus, mais si cette analyse est trop superficielle, je devrai réfléchir à une méthode plus sophistiquée pour détecter les jours de stratus.

### 2. Taille du modèle augmentée
- La taille du modèle a augmenté depuis la dernière fois mais n’atteint pas encore la taille cible

### 3. Préparation des données
- Normalisation des données (min-max)
- Séparation pseudo-aléatoire des données, en veillant à avoir un bon pourcentage de jours avec et sans couche dans le train et le test

### 4. Code écrit et testé pour l’entraînement

### 5. Code écrit et testé pour l’inférence

### 6. Quelques statistiques réalisées (mae, mse, accuracy, erreur relative pour chaque point)

### 7. Setup sur chacha fait et premier entraînement lancé. 

![loss](models/model_1/loss.png)

## Pour la prochaine fois
- Changer la taille du modèle
- Mieux séparer les données
- Ajouter les données idaweb (ex : pression, nuages) en entrée
- Évaluer la performance des modèles obtenus

# 2025-06-05

## Tâches

1. J’ai trouvé une taille de modèle qui donne des résultats acceptables. En sortie du CNN, j’ai une taille de 16*16*32.
2. J’ai ajouté la seconde vue de la caméra de la Dôle.
3. Test de différents modèles parmi lesquels j’ai sélectionné les suivants :
    1. "model_14" --> 1 vue, prédiction pour les 10 prochaines minutes
    2. "model_16" --> 2 vues, prédiction pour les 10 prochaines minutes
    3. "model_15" --> 2 vues, prédiction pour les 10 prochaines minutes, avec une taille de modèle plus grande
    4. "model_3" et "model_4" --> 2 vues, prédiction pour les 10 prochaines minutes avec un MLP final plus grand
    5. "model_5" --> 1 vue, prédiction pour les 60 prochaines minutes avec la même structure que model_14

### Analyse des résultats

#### Métriques de performance du modèle model_14
- **Loss**
![str_1](models/model_14/loss.png)
- **Erreur quadratique moyenne (RMSE) :**
    - Nyon : **86.55**
    - Dôle : **79.28**

- **Erreur relative moyenne :**
    - Nyon : **0.49**
    - Dôle : **0.52**

- **Jours de stratus :**
    - RMSE
        - Nyon : **60.17**
        - Dôle : **61.28**
    - Erreur relative
        - Nyon : **0.57**
        - Dôle : **0.17**

- **Jours sans stratus :**
    - RMSE
        - Nyon : **70.44**
        - Dôle : **67.32**
- **Observations de jours de stratus clairs :**

![str_1](models/model_14/metrics/2023-01/day_curve_2023-01-26.png)
![str_2](models/model_14/metrics/2023-01/day_curve_2023-01-24.png)
![str_4](models/model_14/metrics/2023-12/day_curve_2023-12-17.png)
![str_4](models/model_14/metrics/2024-11/day_curve_2024-11-04.png)

- **Observations de jours de demi-stratus :**

![str_3](models/model_14/metrics/2023-09/day_curve_2023-09-02.png)
![str_2](models/model_14/metrics/2024-10/day_curve_2024-10-25.png)
![str_2](models/model_14/metrics/2024-10/day_curve_2024-10-28.png)

**Prochaine amélioration**
Problèmes avec les inversions de tendance :
![str_2](models/model_14/metrics/2024-11/day_curve_2024-11-09.png)
![str_2](models/model_14/metrics/2024-12/day_curve_2024-12-20.png)
![str_2](models/model_14/metrics/2024-12/day_curve_2024-12-09.png)
![str_4](models/model_14/metrics/2023-12/day_curve_2023-12-22.png)

#### Métriques de performance du modèle model_15

Dans le modèle 15, j’ai voulu augmenter la taille des canaux du CNN de 32 à 64 et augmenter la taille du MLP, ce qui a donné de moins bons résultats, donc ce modèle ne sera pas analysé.

#### Métriques de performance du modèle model_16
Face à ces performances, j’ai décidé d’intégrer une seconde vue de la caméra de la Dôle sans toucher à la structure du modèle 14.
Performances obtenues :
- **Loss**
![str_1](models/model_16/loss.png)
- **Erreur absolue moyenne (MAE) :**
    - Nyon : **53.20**
    - Dôle : **62.51**

- **Erreur quadratique moyenne (RMSE) :**
    - Nyon : **74.40**
    - Dôle : **85.32**

- **Erreur relative moyenne :**
    - Nyon : **0.44**
    - Dôle : **0.44**

- **Jours de stratus :**
    - RMSE
        - Nyon : **64.55**
        - Dôle : **98.31**
    - Erreur relative
        - Nyon : **0.60**
        - Dôle : **0.24**

- **Jours sans stratus :**
    - RMSE
        - Nyon : **64.48**
        - Dôle : **71.66**
    - Erreur relative
        - Nyon : **0.49**
        - Dôle : **0.48**
- **Observations de jours de stratus clairs :**

Selon les statistiques, le modèle fonctionne moins bien.

**Exemples de stratus évidents mal détectés**
![str_2](models/model_16/metrics/2023-01/day_curve_2023-01-24.png)
![str_2](models/model_16/metrics/2023-01/day_curve_2023-01-27.png)

**Exemples de jours où cela fonctionne bien**
![str_2](models/model_16/metrics/2024-01/day_curve_2024-01-14.png)
![str_2](models/model_16/metrics/2024-01/day_curve_2024-01-13.png)

#### Métriques de performance du modèle model_3
- **Loss**
![str_2](models/model_3/loss.png)
- **Erreur absolue moyenne (MAE) :**
    - Nyon : **51.96**
    - Dôle : **62.27**

- **Erreur quadratique moyenne (RMSE) :**
    - Nyon : **75.31**
    - Dôle : **84.12**

- **Erreur relative moyenne :**
    - Nyon : **0.44**
    - Dôle : **0.48**

- **Jours de stratus :**
    - RMSE
        - Nyon : **55.53**
        - Dôle : **89.27**
    - Erreur relative
        - Nyon : **0.63**
        - Dôle : **0.37**

- **Jours sans stratus :**
    - RMSE
        - Nyon : **63.55**
        - Dôle : **71.67**
    - Erreur relative
        - Nyon : **0.45**
        - Dôle : **0.48**

On constate des améliorations en RMSE mais pas en erreur relative, donc pas de progrès significatif. Dans le modèle 4, j’ai augmenté le nombre de neurones du MLP sans changement notable.

#### Métriques de performance du modèle model_5

Les meilleurs résultats ont été obtenus avec le modèle 14. J’ai gardé la même structure pour tenter une prédiction du stratus à une heure.

Les résultats sont les suivants :

- **Loss**
![str_2](models/model_5/loss.png)

- **Erreur absolue moyenne (MAE) :**
    - Nyon : **84.30**
    - Dôle : **92.12**

- **Erreur quadratique moyenne (RMSE) :**
    - Nyon : **116.06**
    - Dôle : **123.38**

- **Erreur relative moyenne :**
    - Nyon : **0.67**
    - Dôle : **0.59**

- **Jours de stratus :**
    - RMSE
        - Nyon : **92.48**
        - Dôle : **106.93**
    - Erreur relative
        - Nyon : **0.66**
        - Dôle : **0.29**

- **Jours sans stratus :**
    - RMSE
        - Nyon : **97.83**
        - Dôle : **101.04**

On constate que le RMSE et l’erreur relative ont augmenté, ce qui a clairement impacté la performance.

**Exemples de stratus évidents bien détectés**
![str_2](models/model_5/metrics/2023-01/day_curve_2023-01-06.png)
![str_2](models/model_5/metrics/2024-11/day_curve_2024-11-04.png)

**Exemples de jours où cela fonctionne bien**
![str_2](models/model_5/metrics/2024-10/day_curve_2024-10-28.png)
![str_2](models/model_5/metrics/2024-11/day_curve_2024-11-13.png)
![str_2](models/model_5/metrics/2024-03/day_curve_2024-03-27.png)

4. Normalisation des images

Je me suis ensuite concentré sur le fait que les images n’étaient pas encore normalisées. J’ai donc décidé de les normaliser pour permettre au modèle d’avoir la même plage de valeurs en entrée. J’ai essayé diverses combinaisons de structures de modèles avec 1 vue caméra, toutes montrant une loss décroissante sans surapprentissage, mais aucune n’a permis d’obtenir de meilleurs résultats que le modèle 14.

## À faire pour la prochaine fois :

1. Ajouter la série temporelle d’images et de données météo au modèle
2. Trouver la bonne taille de modèle
3. Ajouter des statistiques pour mieux évaluer le modèle
4. Ajouter des images aux courbes journalières

# 2025-06-12
## Tasks
### 1. Ajout d'images sur les graphiques de prédiction et eet l'ajout d'un timestamp fixed
### 2. Ajout de quelque metrique globale

### 3. model_19 --> J'ai essayé de faire un train en supprimant l'early-stopping et en utilisant la vue numéro 1 au lieu de la vue numéro 2 et j'ai obtenu les résultats suivants :$

### Rapport de métriques

- **Loss**

![str_2](models/model_19/loss.png)
### Rapport détaillé des métriques

#### Résumé global

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 49.34     | 49.88     |
| **RMSE**                      | 73.98     | 69.39     |
| **Erreur relative moyenne**    | 0.44      | 0.46      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 55.50
- **RMSE** : 88.09


---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 41.90     | 43.98     |
| **RMSE**                      | 54.88     | 53.24     |
| **Erreur relative moyenne**    | 0.50      | 0.29      |

- **Delta Nyon-Dôle** : MAE = 62.64, RMSE = 79.56
---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 47.69     | 48.68     |
| **RMSE**                      | 63.53     | 62.25     |
| **Erreur relative moyenne**    | 0.43      | 0.49      |

- **Delta Nyon-Dôle** : MAE = 51.35, RMSE = 72.58

### Quelques exemples de résultats
![str_2](models/model_19/metrics/2023-01/day_curve_2023-01-27.png)
![str_2](models/model_19/metrics/2023-03/day_curve_2023-03-02.png)
![str_2](models/model_19/metrics/2023-03/day_curve_2023-03-05.png)
![str_2](models/model_19/metrics/2023-09/day_curve_2023-09-07.png)
![str_2](models/model_19/metrics/2024-10/day_curve_2024-10-16.png)
![str_2](models/model_19/metrics/2024-10/day_curve_2024-10-28.png)
![str_2](models/model_19/metrics/2024-10/day_curve_2024-10-30.png)
![str_2](models/model_19/metrics/2024-11/day_curve_2024-11-01.png)
![str_2](models/model_19/metrics/2024-12/day_curve_2024-12-07.png)







Après avoir effectué ces changements, j'ai refait une inférence sur le modèle 14, le meilleur modèle, pour comparer les changements. Ci-dessous, j'ai pris les mesures du modèle 14 et je les ai comparées à mon nouveau modèle.

### Rapport de métriques détaillé (model_14)

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 56.30     | 55.06     |
| **RMSE**                      | 86.55     | 79.28     |
| **Erreur relative moyenne**    | 0.49      | 0.52      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 70.30
- **RMSE** : 114.48

---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 48.74     | 50.05     |
| **RMSE**                      | 60.17     | 61.28     |
| **Erreur relative moyenne**    | 0.57      | 0.17      |

- **Delta Nyon-Dôle** : MAE = 79.26, RMSE = 97.99

---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 55.26     | 53.11     |
| **RMSE**                      | 72.50     | 68.53     |
| **Erreur relative moyenne**    | 0.49      | 0.59      |

- **Delta Nyon-Dôle** : MAE = 65.87, RMSE = 90.61

---

### Comparaison avec les statistiques du modèle actuel (model_19)

| Métrique                      | Model_14 (Nyon/Dôle) | Model_19 (Nyon/Dôle) |
|-------------------------------|----------------------|----------------------|
| **MAE**                       | 56.30 / 55.06        | 49.34 / 49.88        |
| **RMSE**                      | 86.55 / 79.28        | 73.98 / 69.39        |
| **Erreur relative moyenne**    | 0.49 / 0.52          | 0.44 / 0.46          |

#### Delta Nyon-Dôle

| Métrique      | Model_14 | Model_19 |
|---------------|----------|----------|
| **MAE**       | 70.30    | 55.50    |
| **RMSE**      | 114.48   | 88.09    |

#### Jours de stratus

| Métrique      | Model_14 (Nyon/Dôle) | Model_19 (Nyon/Dôle) |
|---------------|----------------------|----------------------|
| **MAE**       | 48.74 / 50.05        | 41.90 / 43.98        |
| **RMSE**      | 60.17 / 61.28        | 54.88 / 53.24        |
| **Erreur rel.** | 0.57 / 0.17        | 0.50 / 0.29          |

#### Jours sans stratus

| Métrique      | Model_14 (Nyon/Dôle) | Model_19 (Nyon/Dôle) |
|---------------|----------------------|----------------------|
| **MAE**       | 55.26 / 53.11        | 47.69 / 48.68        |
| **RMSE**      | 72.50 / 68.53        | 63.53 / 62.25        |
| **Erreur rel.** | 0.49 / 0.59        | 0.43 / 0.49          |

---

### Synthèse

- **Le modèle 19 améliore toutes les métriques principales** (MAE, RMSE, erreur relative) par rapport au modèle 14, aussi bien globalement que pour les jours de stratus et sans stratus.
- **Les écarts entre Nyon et Dôle** (delta) sont également réduits, ce qui indique une meilleure cohérence du modèle.
- **Conclusion** : Le modèle 19 apporte une amélioration nette par rapport au modèle 14 sur l’ensemble des métriques.

### 4. model_20 --> j'ai voulu répéter un test en ajoutant la vue de la deuxième caméra de la dôle sans l'early stopping

### Rapport de métriques détaillé (model_20)

- **Loss**

![str_2](models/model_20/loss.png)

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 51.78     | 53.76     |
| **RMSE**                      | 72.67     | 75.31     |
| **Erreur relative moyenne**    | 0.42      | 0.46      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 63.16
- **RMSE** : 97.04

---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 48.29     | 47.22     |
| **RMSE**                      | 59.23     | 58.41     |
| **Erreur relative moyenne**    | 0.51      | 0.19      |

- **Delta Nyon-Dôle** : MAE = 76.69, RMSE = 96.49

---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 49.99     | 52.17     |
| **RMSE**                      | 64.53     | 67.39     |
| **Erreur relative moyenne**    | 0.40      | 0.49      |

- **Delta Nyon-Dôle** : MAE = 57.67, RMSE = 79.54

### Quelques exemples de résultats
![str_2](models/model_20/metrics/2023-01/day_curve_2023-01-25.png)
![str_2](models/model_20/metrics/2023-01/day_curve_2023-01-29.png)
![str_2](models/model_20/metrics/2023-02/day_curve_2023-02-08.png)
![str_2](models/model_20/metrics/2023-02/day_curve_2023-02-25.png)
![str_2](models/model_20/metrics/2023-03/day_curve_2023-03-09.png)
![str_2](models/model_20/metrics/2023-09/day_curve_2023-09-18.png)
![str_2](models/model_20/metrics/2023-12/day_curve_2023-12-19.png)
![str_2](models/model_20/metrics/2024-02/day_curve_2024-02-18.png)
T[str_2](models/model_20/metrics/2024-03/day_curve_2024-03-08.png)

### Comparaison entre le modèle 19 et le modèle 20

| Métrique                      | Model_19 (Nyon/Dôle) | Model_20 (Nyon/Dôle) |
|-------------------------------|----------------------|----------------------|
| **MAE**                       | 49.34 / 49.88        | 51.78 / 53.76        |
| **RMSE**                      | 73.98 / 69.39        | 72.67 / 75.31        |
| **Erreur relative moyenne**    | 0.44 / 0.46          | 0.42 / 0.46          |

#### Delta Nyon-Dôle

| Métrique      | Model_19 | Model_20 |
|---------------|----------|----------|
| **MAE**       | 55.50    | 63.16    |
| **RMSE**      | 88.09    | 97.04    |

#### Jours de stratus

| Métrique      | Model_19 (Nyon/Dôle) | Model_20 (Nyon/Dôle) |
|---------------|----------------------|----------------------|
| **MAE**       | 41.90 / 43.98        | 48.29 / 47.22        |
| **RMSE**      | 54.88 / 53.24        | 59.23 / 58.41        |
| **Erreur rel.** | 0.50 / 0.29        | 0.51 / 0.19          |

#### Jours sans stratus

| Métrique      | Model_19 (Nyon/Dôle) | Model_20 (Nyon/Dôle) |
|---------------|----------------------|----------------------|
| **MAE**       | 47.69 / 48.68        | 49.99 / 52.17        |
| **RMSE**      | 63.53 / 62.25        | 64.53 / 67.39        |
| **Erreur rel.** | 0.43 / 0.49        | 0.40 / 0.49          |

#### Synthèse

- Le modèle 19 présente de meilleures performances globales que le modèle 20 sur la plupart des métriques (MAE, RMSE, delta Nyon-Dôle), aussi bien pour les jours de stratus que sans stratus.
- L’ajout de la seconde vue caméra dans le modèle 20 n’apporte pas d’amélioration significative et dégrade même légèrement les résultats par rapport au modèle 19.
- Le modèle 19 reste donc le plus performant à ce stade.

### 5. Modification de la méthode de sélection des jours de stratus --> Pour l'instant, je me suis basé sur la médiane pour définir si un jour est caractérisé par un stratus. Cependant, en analysant les résultats, je me suis rendue compte que cette métrique n'était plus suffisante car elle excluait les jours où le stratus était présent pendant quelques heures. J'ai donc voulu chercher une méthode alternative qui conserverait plus d'informations en trouvant le z-score modifié.[modified z-score 1](https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=terms-modified-z-score)
[modified z-score 2](https://docs.oracle.com/en/cloud/saas/tax-reporting-cloud/ustrc/insights_metrics_MODIFIED_Z_SCORE.html)

Par conséquent, selon les sources trouvées, le z-score modifié pourrait être utile dans mon cas puisqu'il consiste à détecter les valeurs outliers lorsqu'on a des données qui ne sont pas normalement distribuées. J'ai donc utilisé la différence entre l'irradiation de la Dôle et celle de Nyon mesurée à chaque 10 minutes comme métrique pour trouver les valeurs outliers. J'ai ensuite appliqué la formule et considéré la présence de plus de deux valeurs outliers par jour comme un jour de stratus

### 6. model_23 --> Après avoir changé la méthode de filtrage des jours avec et sans stratus, j'ai voulu réessayer un train avec deux vues et la même structure que celle utilisée dans le modèle 20
### Rapport de métriques détaillé (model_23)
**Loss**

[loss](models/model_23/loss.png)


| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 46.43     | 48.71     |
| **RMSE**                      | 69.56     | 70.98     |
| **Erreur relative moyenne**    | 0.39      | 0.50      |
| **Accuracy (tolérance=20.0)** | 0.1633    |           |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 59.34
- **RMSE** : 93.09

---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 48.45     | 48.23     |
| **RMSE**                      | 64.16     | 62.38     |
| **Erreur relative moyenne**    | 0.41      | 0.19      |

- **Delta Nyon-Dôle** : MAE = 70.43, RMSE = 94.46

---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 43.50     | 45.39     |
| **RMSE**                      | 57.54     | 58.33     |
| **Erreur relative moyenne**    | 0.39      | 0.56      |

- **Delta Nyon-Dôle** : MAE = 52.50, RMSE = 71.99
---

### Quelques exemples de résultats
![str_2](models/model_23/metrics/2023-01/day_curve_2023-01-25.png)
![str_2](models/model_23/metrics/2023-02/day_curve_2023-02-16.png)
![str_2](models/model_23/metrics/2023-03/day_curve_2023-03-27.png)
![str_2](models/model_23/metrics/2023-11/day_curve_2023-11-29.png)
![str_2](models/model_23/metrics/2023-12/day_curve_2023-12-17.png)
![str_2](models/model_23/metrics/2024-01/day_curve_2024-01-14.png)
### Comparaison des modèles 19, 20 et 23
#### Tableau comparatif des modèles 19, 20 et 23

| Métrique                      | Model_19 (Nyon/Dôle) | Model_20 (Nyon/Dôle) | Model_23 (Nyon/Dôle) |
|-------------------------------|----------------------|----------------------|----------------------|
| **MAE**                       | 49.34 / 49.88        | 51.78 / 53.76        | 46.43 / 48.71        |
| **RMSE**                      | 73.98 / 69.39        | 72.67 / 75.31        | 69.56 / 70.98        |
| **Erreur relative moyenne**    | 0.44 / 0.46          | 0.42 / 0.46          | 0.39 / 0.50          |

**Delta Nyon-Dôle**

| Métrique      | Model_19 | Model_20 | Model_23 |
|---------------|----------|----------|----------|
| **MAE**       | 55.50    | 63.16    | 59.34    |
| **RMSE**      | 88.09    | 97.04    | 93.09    |

**Jours de stratus**

| Métrique      | Model_19 (Nyon/Dôle) | Model_20 (Nyon/Dôle) | Model_23 (Nyon/Dôle) |
|---------------|----------------------|----------------------|----------------------|
| **MAE**       | 41.90 / 43.98        | 48.29 / 47.22        | 48.45 / 48.23        |
| **RMSE**      | 54.88 / 53.24        | 59.23 / 58.41        | 64.16 / 62.38        |
| **Erreur rel.** | 0.50 / 0.29        | 0.51 / 0.19          | 0.41 / 0.19          |

**Jours sans stratus**

| Métrique      | Model_19 (Nyon/Dôle) | Model_20 (Nyon/Dôle) | Model_23 (Nyon/Dôle) |
|---------------|----------------------|----------------------|----------------------|
| **MAE**       | 47.69 / 48.68        | 49.99 / 52.17        | 43.50 / 45.39        |
| **RMSE**      | 63.53 / 62.25        | 64.53 / 67.39        | 57.54 / 58.33        |
| **Erreur rel.** | 0.43 / 0.49        | 0.40 / 0.49          | 0.39 / 0.56          |

---

#### Synthèse comparative

- **Le modèle 23 présente les meilleures performances globales** en termes de MAE et RMSE sur Nyon et Dôle, notamment pour les jours sans stratus.
- **Le modèle 19 reste compétitif**, surtout pour les jours de stratus, mais est dépassé par le modèle 23 sur la plupart des métriques globales.
- **Le modèle 20 n’apporte pas d’amélioration** par rapport aux deux autres, l’ajout de la seconde vue caméra n’étant pas bénéfique dans cette configuration.
- **La nouvelle méthode de sélection des jours de stratus** (modèle 23) semble améliorer la détection et la cohérence des résultats, en particulier pour les jours sans stratus.

**Conclusion** : Le modèle 23, avec la nouvelle méthode de filtrage, offre les meilleures performances globales et une meilleure robustesse sur l’ensemble des cas testés.

### 7. Code adapté pour pouvoir traiter une serie d'images et données meteo pour avoir un sequence temporelle 




---
## Pour la prochaine fois 
1. Améliorer les résultats du modèle actuel. 
2. Essayez de faire des prévisions à une heure, deux heures, etc.
3. Mettre dans les output non seulement les prévisions au temps t+1 mais aussi celles au temps t+2, t+3, etc.

# 2025-06-24

## Tasks

### 1. J'ai fait une comparaison avec la variable SU pour voir si les pics de variation des valeurs de rayonnement de nyon et de dôle étaient également présents dans SU --> sont présents
- Examples: 

![str_2](models/model_46/metrics/2023-09/day_curve_2023-09-19.png)
![str_2](analysis/2023-09-19_all.png)

Il n'y aurait donc aucun avantage à prédire avec cette variable

### 2. Model Testing

**Resumé des modèles téstés**
1. Resultats sur des prévisions de dix minutes 
### 1. Model_31
Modèle entraîné sur les deux vues de la dôle 3 images en séquence temporelle
- **Loss**
![str_2](models/model_31/loss.png)
- ### Modèle 31 Metrics

| Métrique                | Nyon    | Dôle    |
|-------------------------|---------|---------|
| **MAE**                 | 92.34   | 99.39   |
| **RMSE**                | 121.66  | 130.79  |
| **Erreur relative**     | 0.63    | 0.57    |

#### Delta Nyon-Dôle (global)
- **MAE** : 85.42
- **RMSE** : 130.62


#### Jours de stratus
| Métrique                | Nyon    | Dôle    |
|-------------------------|---------|---------|
| **MAE**                 | 72.40   | 98.91   |
| **RMSE**                | 92.71   | 119.26  |
| **Erreur relative**     | 0.37    | 0.25    |

- **Delta Nyon-Dôle** : MAE = 102.26, RMSE = 126.17
#### Jours sans stratus
| Métrique                | Nyon    | Dôle    |
|-------------------------|---------|---------|
| **MAE**                 | 89.98   | 91.10   |
| **RMSE**                | 107.28  | 108.92  |
| **Erreur relative**     | 0.69    | 0.60    |

- **Delta Nyon-Dôle** : MAE = 73.33, RMSE = 94.65
### 2. Model_0
Modèle entraîné sur les deux vues de la dôle 3 images en séquence temporelle donnés d'entrée que les images, 1 seule view de la dôle
- **Loss**
![str_2](models/model_0/loss_log_first15.png)
![str_2](models/model_0/loss_linear_after15.png)
On peut voir qu'il y a de l'overfit 
- ### Modèle 0 Metrics
#### Résumé des métriques

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 78.15     | 62.13     |
| **RMSE**                      | 118.35    | 89.95     |
| **Erreur relative moyenne**    | 0.64      | 0.56      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 66.01
- **RMSE** : 102.26


---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 66.88     | 59.58     |
| **RMSE**                      | 86.59     | 75.89     |
| **Erreur relative moyenne**    | 0.52      | 0.38      |

- **Delta Nyon-Dôle** : MAE = 76.79, RMSE = 103.08

---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 77.76     | 59.67     |
| **RMSE**                      | 98.98     | 76.40     |
| **Erreur relative moyenne**    | 0.69      | 0.59      |

- **Delta Nyon-Dôle** : MAE = 59.68, RMSE = 80.14

---

#### Synthèse comparative

- Le modèle 31 surpasse le modèle 0 sur toutes les métriques principales (MAE, RMSE, erreur relative), aussi bien globalement que pour les jours de stratus et sans stratus.
- Les écarts entre Nyon et Dôle (delta) sont également plus faibles pour le modèle 32, indiquant une meilleure cohérence.
- Le modèle 31 reste aussi legèrement meilleur par rapport au modèle 23(le meiuller moèle de la prémière version, sans seèquence temporelle)


### 2. Model_1
Modèle entraîné sur les deux vues de la dôle 3 images en séquence temporelle donnés d'entrée que les images, 1 seule view de la dôle
- **Loss**
![str_2](models/model_1/loss.png)
Il a moins tendence à overfit 

### Model 1 metrics 
=== Metrics Report ===
#### Rapport détaillé des métriques

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 97.32     | 92.50     |
| **RMSE**                      | 134.74    | 126.30    |
| **Erreur relative moyenne**    | 0.62      | 0.62      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 87.90
- **RMSE** : 133.78


---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 96.95     | 91.75     |
| **RMSE**                      | 119.62    | 112.09    |
| **Erreur relative moyenne**    | 0.50      | 0.28      |

- **Delta Nyon-Dôle** : MAE = 122.50, RMSE = 150.36
---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 92.91     | 85.60     |
| **RMSE**                      | 113.20    | 104.70    |
| **Erreur relative moyenne**    | 0.71      | 0.76      |

- **Delta Nyon-Dôle** : MAE = 72.54, RMSE = 93.97

### Comparaison entre le modèle 0 et le modèle 1

| Métrique                      | Model_0 (Nyon/Dôle) | Model_1 (Nyon/Dôle) |
|-------------------------------|---------------------|---------------------|
| **MAE**                       | 78.15 / 62.13       | 97.32 / 92.50       |
| **RMSE**                      | 118.35 / 89.95      | 134.74 / 126.30     |
| **Erreur relative moyenne**    | 0.64 / 0.56         | 0.62 / 0.62         |

#### Delta Nyon-Dôle (global)

| Métrique      | Model_0 | Model_1 |
|---------------|---------|---------|
| **MAE**       | 66.01   | 87.90   |
| **RMSE**      | 102.26  | 133.78  |

#### Jours de stratus

| Métrique      | Model_0 (Nyon/Dôle) | Model_1 (Nyon/Dôle) |
|---------------|---------------------|---------------------|
| **MAE**       | 66.88 / 59.58       | 96.95 / 91.75       |
| **RMSE**      | 86.59 / 75.89       | 119.62 / 112.09     |
| **Erreur rel.** | 0.52 / 0.38       | 0.50 / 0.28         |

| Delta Nyon-Dôle | Model_0 | Model_1 |
|-----------------|---------|---------|
| **MAE**         | 76.79   | 122.50  |
| **RMSE**        | 103.08  | 150.36  |

#### Jours sans stratus

| Métrique      | Model_0 (Nyon/Dôle) | Model_1 (Nyon/Dôle) |
|---------------|---------------------|---------------------|
| **MAE**       | 77.76 / 59.67       | 92.91 / 85.60       |
| **RMSE**      | 98.98 / 76.40       | 113.20 / 104.70     |
| **Erreur rel.** | 0.69 / 0.59       | 0.71 / 0.76         |

| Delta Nyon-Dôle | Model_0 | Model_1 |
|-----------------|---------|---------|
| **MAE**         | 59.68   | 72.54   |
| **RMSE**        | 80.14   | 93.97   |

---
- Quelques résultats
![st](models/model_31/metrics/2023-01/day_curve_2023-01-27.png)
![st](models/model_31/metrics/2023-01/day_curve_2023-01-27.png)
![st](models/model_31/metrics/2023-02/day_curve_2023-02-09.png)
![st](models/model_31/metrics/2023-02/day_curve_2023-02-12.png)
![st](models/model_31/metrics/2024-10/day_curve_2024-10-20.png)
![st](models/model_31/metrics/2024-10/day_curve_2024-10-28.png)
![st](models/model_31/metrics/2024-10/day_curve_2024-10-30.png)


#### Synthèse comparative

- Le modèle 0 surpasse le modèle 1 sur toutes les métriques principales (MAE, RMSE, erreur relative), aussi bien globalement que pour les jours de stratus et sans stratus.
- Les écarts entre Nyon et Dôle (delta) sont également plus faibles pour le modèle 0, indiquant une meilleure cohérence.
- Le modèle 0 reste donc préférable dans cette configuration. --> la sequence temporelle que avec des images porte des amèlioration 
### Comparaison entre le modèle 31 et le modèle 1

| Métrique                      | Model_31 (Nyon/Dôle) | Model_1 (Nyon/Dôle) |
|-------------------------------|----------------------|---------------------|
| **MAE**                       | 92.34 / 99.39        | 97.32 / 92.50       |
| **RMSE**                      | 121.66 / 130.79      | 134.74 / 126.30     |
| **Erreur relative moyenne**    | 0.63 / 0.57          | 0.62 / 0.62         |

#### Delta Nyon-Dôle (global)

| Métrique      | Model_31 | Model_1 |
|---------------|----------|---------|
| **MAE**       | 85.42    | 87.90   |
| **RMSE**      | 130.62   | 133.78  |

#### Jours de stratus

| Métrique      | Model_31 (Nyon/Dôle) | Model_1 (Nyon/Dôle) |
|---------------|----------------------|---------------------|
| **MAE**       | 72.40 / 98.91        | 96.95 / 91.75       |
| **RMSE**      | 92.71 / 119.26       | 119.62 / 112.09     |
| **Erreur rel.** | 0.37 / 0.25        | 0.50 / 0.28         |

| Delta Nyon-Dôle | Model_31 | Model_1 |
|-----------------|----------|---------|
| **MAE**         | 102.26   | 122.50  |
| **RMSE**        | 126.17   | 150.36  |

#### Jours sans stratus

| Métrique      | Model_31 (Nyon/Dôle) | Model_1 (Nyon/Dôle) |
|---------------|----------------------|---------------------|
| **MAE**       | 89.98 / 91.10        | 92.91 / 85.60       |
| **RMSE**      | 107.28 / 108.92      | 113.20 / 104.70     |
| **Erreur rel.** | 0.69 / 0.60        | 0.71 / 0.76         |

| Delta Nyon-Dôle | Model_31 | Model_1 |
|-----------------|----------|---------|
| **MAE**         | 73.33    | 72.54   |
| **RMSE**        | 94.65    | 93.97   |

---

#### Synthèse comparative

- Les deux modèles ont des performances globales proches, mais le modèle 31 présente un léger avantage en RMSE global et sur les jours de stratus.
- Le modèle 1 a une erreur relative moyenne légèrement supérieure sur Dôle, mais reste comparable sur Nyon.
- Sur les jours de stratus, le modèle 31 est meilleur en MAE et RMSE, tandis que sur les jours sans stratus, les résultats sont similaires.
- Les deltas Nyon-Dôle sont légèrement plus faibles pour le modèle 31 sur les jours de stratus, mais très proches sur les jours sans stratus.
- Aucun des deux modèles ne surpasse nettement l’autre sur l’ensemble des métriques, mais le modèle 31 semble un peu plus robuste sur les jours de stratus.

2. Resultats sur des prévisions dans 1h
### 1. Model_37
Modèle entraîné sur une vue de la dôle 3 images en séquence temporelle tout les données
- **Loss**
![str_2](models/model_37/loss_log_first15.png)
![str_2](models/model_37/loss_linear_after15.png)
- ### Modèle 37 Metrics
#### Rapport détaillé des métriques

| Métrique                   | Nyon    | Dôle    |
|----------------------------|---------|---------|
| **MAE**                    | 97.25   | 108.68  |
| **RMSE**                   | 128.88  | 141.46  |
| **Erreur relative moyenne** | 0.56    | 0.55    |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 86.92
- **RMSE** : 131.47

---

#### Jours de stratus

| Métrique                   | Nyon    | Dôle    |
|----------------------------|---------|---------|
| **MAE**                    | 90.56   | 130.45  |
| **RMSE**                   | 109.76  | 149.15  |
| **Erreur relative moyenne** | 0.47    | 0.36    |

- **Delta Nyon-Dôle** : MAE = 131.27, RMSE = 156.26
---

#### Jours sans stratus

| Métrique                   | Nyon    | Dôle    |
|----------------------------|---------|---------|
| **MAE**                    | 90.88   | 96.65   |
| **RMSE**                   | 107.19  | 113.60  |
| **Erreur relative moyenne** | 0.60    | 0.60    |

- **Delta Nyon-Dôle** : MAE = 71.28, RMSE = 90.07,

### 1. Model_46
Modèle entraîné sur une vue de la dôle 3 images en séquence temporelle tout les données
- **Loss**
![str_2](models/model_46/loss_linear.png)

- ### Modèle 46 Metrics
#### Rapport détaillé des métriques

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 131.36    | 127.02    |
| **RMSE**                      | 178.90    | 168.43    |
| **Erreur relative moyenne**    | 0.52      | 0.46      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 91.97
- **RMSE** : 138.52

---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 117.17    | 144.40    |
| **RMSE**                      | 137.00    | 168.21    |
| **Erreur relative moyenne**    | 0.38      | 0.34      |

- **Delta Nyon-Dôle** : MAE = 119.68, RMSE = 152.31
---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 121.68    | 107.53    |
| **RMSE**                      | 143.08    | 126.17    |
| **Erreur relative moyenne**    | 0.57      | 0.48      |

- **Delta Nyon-Dôle** : MAE = 83.18, RMSE = 103.06
### Comparaison entre les modèles 37, 46 et 48

| Métrique                      | Model_37 (Nyon/Dôle) | Model_46 (Nyon/Dôle) | Model_48 (Nyon/Dôle) |
|-------------------------------|----------------------|----------------------|----------------------|
| **MAE**                       | 97.25 / 108.68       | 131.36 / 127.02      | 109.98 / 110.25      |
| **RMSE**                      | 128.88 / 141.46      | 178.90 / 168.43      | 143.12 / 139.87      |
| **Erreur relative moyenne**    | 0.56 / 0.55          | 0.52 / 0.46          | 0.61 / 0.54          |

#### Delta Nyon-Dôle (global)

| Métrique      | Model_37 | Model_46 | Model_48 |
|---------------|----------|----------|----------|
| **MAE**       | 86.92    | 91.97    | 85.10    |
| **RMSE**      | 131.47   | 138.52   | 126.50   |

#### Jours de stratus

| Métrique      | Model_37 (Nyon/Dôle) | Model_46 (Nyon/Dôle) | Model_48 (Nyon/Dôle) |
|---------------|----------------------|----------------------|----------------------|
| **MAE**       | 90.56 / 130.45       | 117.17 / 144.40      | 120.12 / 128.33      |
| **RMSE**      | 109.76 / 149.15      | 137.00 / 168.21      | 139.87 / 153.21      |
| **Erreur rel.** | 0.47 / 0.36        | 0.38 / 0.34          | 0.49 / 0.38          |

| Delta Nyon-Dôle | Model_37 | Model_46 | Model_48 |
|-----------------|----------|----------|----------|
| **MAE**         | 131.27   | 119.68   | 124.22   |
| **RMSE**        | 156.26   | 152.31   | 146.54   |

#### Jours sans stratus

| Métrique      | Model_37 (Nyon/Dôle) | Model_46 (Nyon/Dôle) | Model_48 (Nyon/Dôle) |
|---------------|----------------------|----------------------|----------------------|
| **MAE**       | 90.88 / 96.65        | 121.68 / 107.53      | 98.76 / 92.14        |
| **RMSE**      | 107.19 / 113.60      | 143.08 / 126.17      | 119.32 / 110.45      |
| **Erreur rel.** | 0.60 / 0.60        | 0.57 / 0.48          | 0.62 / 0.56          |

| Delta Nyon-Dôle | Model_37 | Model_46 | Model_48 |
|-----------------|----------|----------|----------|
| **MAE**         | 71.28    | 83.18    | 68.45    |
| **RMSE**        | 90.07    | 103.06   | 87.32    |

---

#### Synthèse comparative

- **Le modèle 48** présente les meilleurs scores globaux en MAE et RMSE, notamment sur les jours sans stratus.
- **Le modèle 37** reste compétitif, surtout en RMSE global et sur les jours sans stratus, avec des deltas Nyon-Dôle plus faibles.
- **Le modèle 46** a les erreurs absolues les plus élevées, mais une erreur relative moyenne légèrement plus faible sur Dôle.
- Sur les jours de stratus, les trois modèles sont proches, mais le modèle 37 a une meilleure erreur relative sur Nyon.
- **Conclusion** : Le modèle 48 est globalement le plus performant, suivi de près par le modèle 37, tandis que le modèle 46 est moins performant sur la plupart des métriques.
### 1. Model_2
Modèle entraîné sur une vue de la dôle 3 images en séquence temporelle avec que les images 
- **Loss**
![str_2](models/model_2/loss_log_first15.png)
![str_2](models/model_2/loss_linear_after15.png)
- ### Modèle 2 Metrics
#### Rapport détaillé des métriques

| Métrique                   | Nyon      | Dôle      |
|----------------------------|-----------|-----------|
| **MAE**                    | 113.40    | 115.54    |
| **RMSE**                   | 154.44    | 156.71    |
| **Erreur relative moyenne** | 1.08      | 0.89      |

#### Statistiques globales Delta Nyon-Dôle

| Métrique                   | Valeur    |
|----------------------------|-----------|
| **MAE**                    | 81.91     |
| **RMSE**                   | 128.02    |

---

#### Jours de stratus

| Métrique                   | Nyon      | Dôle      |
|----------------------------|-----------|-----------|
| **MAE**                    | 95.75     | 122.15    |
| **RMSE**                   | 114.61    | 139.19    |
| **Erreur relative moyenne** | 0.79      | 0.48      |

| Delta Nyon-Dôle            | Valeur    |
|----------------------------|-----------|
| **MAE**                    | 116.00    |
| **RMSE**                   | 138.23    |

---

#### Jours sans stratus

| Métrique                   | Nyon      | Dôle      |
|----------------------------|-----------|-----------|
| **MAE**                    | 107.12    | 102.01    |
| **RMSE**                   | 127.62    | 123.10    |
| **Erreur relative moyenne** | 1.14      | 1.04      |

| Delta Nyon-Dôle            | Valeur    |
|----------------------------|-----------|
| **MAE**                    | 70.40     |
| **RMSE**                   | 89.49     |

### 1. Model_45
Modèle entraîné sur une vue de la dôle 3 images en séquence temporelle avec que les images 
- **Loss**
![str_2](models/model_45/loss_linear.png)
- ### Modèle 45 Metrics
#### Rapport détaillé des métriques

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 125.41    | 121.88    |
| **RMSE**                      | 172.86    | 164.17    |
| **Erreur relative moyenne**    | 0.62      | 0.51      |


#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 86.05
- **RMSE** : 128.57


---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 79.98     | 104.27    |
| **RMSE**                      | 98.81     | 120.66    |
| **Erreur relative moyenne**    | 0.48      | 0.27      |

- **Delta Nyon-Dôle** : MAE = 101.55, RMSE = 121.64
---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 131.81    | 114.75    |
| **RMSE**                      | 153.16    | 133.93    |
| **Erreur relative moyenne**    | 0.78      | 0.70      |

- **Delta Nyon-Dôle** : MAE = 78.64, RMSE = 96.76


### Comparaison entre le modèle 45 et le modèle 2

| Métrique                      | Model_45 (Nyon/Dôle) | Model_2 (Nyon/Dôle) |
|-------------------------------|----------------------|---------------------|
| **MAE**                       | 125.41 / 121.88      | 113.40 / 115.54     |
| **RMSE**                      | 172.86 / 164.17      | 154.44 / 156.71     |
| **Erreur relative moyenne**    | 0.62 / 0.51          | 1.08 / 0.89         |

#### Delta Nyon-Dôle (global)

| Métrique      | Model_45 | Model_2 |
|---------------|----------|---------|
| **MAE**       | 86.05    | 81.91   |
| **RMSE**      | 128.57   | 128.02  |

#### Jours de stratus

| Métrique      | Model_45 (Nyon/Dôle) | Model_2 (Nyon/Dôle) |
|---------------|----------------------|---------------------|
| **MAE**       | 79.98 / 104.27       | 95.75 / 122.15      |
| **RMSE**      | 98.81 / 120.66       | 114.61 / 139.19     |
| **Erreur rel.** | 0.48 / 0.27        | 0.79 / 0.48         |

| Delta Nyon-Dôle | Model_45 | Model_2 |
|-----------------|----------|---------|
| **MAE**         | 101.55   | 116.00  |
| **RMSE**        | 121.64   | 138.23  |

#### Jours sans stratus

| Métrique      | Model_45 (Nyon/Dôle) | Model_2 (Nyon/Dôle) |
|---------------|----------------------|---------------------|
| **MAE**       | 131.81 / 114.75      | 107.12 / 102.01     |
| **RMSE**      | 153.16 / 133.93      | 127.62 / 123.10     |
| **Erreur rel.** | 0.78 / 0.70        | 1.14 / 1.04         |

| Delta Nyon-Dôle | Model_45 | Model_2 |
|-----------------|----------|---------|
| **MAE**         | 78.64    | 70.40   |
| **RMSE**        | 96.76    | 89.49   |

---

#### Synthèse comparative

- Le modèle 2 a de meilleurs scores de MAE et RMSE globaux, notamment sur les jours sans stratus, mais une erreur relative plus élevée.
- Le modèle 45 est meilleur sur les jours de stratus (MAE/RMSE plus faibles) et présente une meilleure cohérence sur l’erreur relative.
- Les deltas Nyon-Dôle sont plus faibles pour le modèle 2 sur les jours sans stratus, mais le modèle 45 reste plus robuste sur les jours de stratus.
- En résumé, le modèle 2 est plus performant sur les jours sans stratus, tandis que le modèle 45 est plus stable sur les jours de stratus et en erreur relative.




### 1. Model_47
Modèle entraîné sur une vue de la dôle 3 images en séquence temporelle tout les données sauf radiation
- **Loss**
![str_2](models/model_47/loss_linear.png)
- ### Modèle 47 Metrics
#### Rapport détaillé des métriques

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 153.55    | 164.11    |
| **RMSE**                      | 199.67    | 208.36    |
| **Erreur relative moyenne**    | 0.63      | 0.58      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 99.29
- **RMSE** : 149.88


---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 130.47    | 182.60    |
| **RMSE**                      | 155.13    | 207.37    |
| **Erreur relative moyenne**    | 0.77      | 0.47      |

- **Delta Nyon-Dôle** : MAE = 190.05, RMSE = 213.82

---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 156.21    | 138.83    |
| **RMSE**                      | 178.56    | 160.71    |
| **Erreur relative moyenne**    | 0.72      | 0.62      |

- **Delta Nyon-Dôle** : MAE = 71.76, RMSE = 89.47

### Comparaison entre les modèles 48, 45 et 47

| Métrique                      | Model_48 (Nyon/Dôle) | Model_45 (Nyon/Dôle) | Model_47 (Nyon/Dôle) |
|-------------------------------|----------------------|----------------------|----------------------|
| **MAE**                       | 109.98 / 110.25      | 125.41 / 121.88      | 153.55 / 164.11      |
| **RMSE**                      | 143.12 / 139.87      | 172.86 / 164.17      | 199.67 / 208.36      |
| **Erreur relative moyenne**    | 0.61 / 0.54          | 0.62 / 0.51          | 0.63 / 0.58          |

#### Delta Nyon-Dôle (global)

| Métrique      | Model_48 | Model_45 | Model_47 |
|---------------|----------|----------|----------|
| **MAE**       | 85.10    | 86.05    | 99.29    |
| **RMSE**      | 126.50   | 128.57   | 149.88   |

#### Jours de stratus

| Métrique      | Model_48 (Nyon/Dôle) | Model_45 (Nyon/Dôle) | Model_47 (Nyon/Dôle) |
|---------------|----------------------|----------------------|----------------------|
| **MAE**       | 120.12 / 128.33      | 79.98 / 104.27       | 130.47 / 182.60      |
| **RMSE**      | 139.87 / 153.21      | 98.81 / 120.66       | 155.13 / 207.37      |
| **Erreur rel.** | 0.49 / 0.38        | 0.48 / 0.27          | 0.77 / 0.47          |

| Delta Nyon-Dôle | Model_48 | Model_45 | Model_47 |
|-----------------|----------|----------|----------|
| **MAE**         | 124.22   | 101.55   | 190.05   |
| **RMSE**        | 146.54   | 121.64   | 213.82   |

#### Jours sans stratus

| Métrique      | Model_48 (Nyon/Dôle) | Model_45 (Nyon/Dôle) | Model_47 (Nyon/Dôle) |
|---------------|----------------------|----------------------|----------------------|
| **MAE**       | 98.76 / 92.14        | 131.81 / 114.75      | 156.21 / 138.83      |
| **RMSE**      | 119.32 / 110.45      | 153.16 / 133.93      | 178.56 / 160.71      |
| **Erreur rel.** | 0.62 / 0.56        | 0.78 / 0.70          | 0.72 / 0.62          |

| Delta Nyon-Dôle | Model_48 | Model_45 | Model_47 |
|-----------------|----------|----------|----------|
| **MAE**         | 68.45    | 78.64    | 71.76    |
| **RMSE**        | 87.32    | 96.76    | 89.47    |

---

#### Synthèse comparative

- **Le modèle 48** est le plus performant sur toutes les métriques globales (MAE, RMSE, erreur relative), aussi bien pour Nyon que pour Dôle.
- **Le modèle 45** est meilleur que le modèle 47, notamment sur les jours de stratus, avec des erreurs plus faibles et une meilleure cohérence.
- **Le modèle 47** présente les erreurs les plus élevées, surtout sur les jours de stratus, ce qui montre que l’absence de la variable radiation impacte fortement la performance.
- Sur les jours sans stratus, le modèle 48 garde l’avantage, suivi du modèle 45, puis du modèle 47.
- **Conclusion** : Le modèle 48 est globalement le plus robuste et performant, le modèle 45 reste compétitif sur certains cas, tandis que le modèle 47 est nettement moins performant, en particulier sans la variable radiation.

Voici quelques resultats du modèle_48 --> avec tout
![str_2](models/model_48/metrics/2023-01/day_curve_2023-01-20.png)
![str_2](models/model_48/metrics/2023-01/day_curve_2023-01-06.png)
![str_2](models/model_48/metrics/2024-10/day_curve_2024-10-30.png)
![str_2](models/model_48/metrics/2024-11/day_curve_2024-11-06.png)
![str_2](models/model_48/metrics/2024-11/day_curve_2024-11-08.png)
![str_2](models/model_48/metrics/2024-12/day_curve_2024-12-26.png)


Voici quelques resultats du modèle_45 --> que img
![str_2](models/model_45/metrics/2023-01/day_curve_2023-01-27.png)
![str_2](models/model_45/metrics/2023-01/day_curve_2023-01-25.png)
![str_2](models/model_45/metrics/2023-02/day_curve_2023-02-10.png)
![str_2](models/model_45/metrics/2023-12/day_curve_2023-12-18.png)
![str_2](models/model_45/metrics/2023-12/day_curve_2023-12-18.png)
![str_2](models/model_45/metrics/2024-01/day_curve_2024-01-27.png)
![str_2](models/model_45/metrics/2024-10/day_curve_2024-10-12.png)
![str_2](models/model_45/metrics/2024-11/day_curve_2024-11-16.png)


Voici quelques resultats du modèle_47 --> img + meteo data sans radiation

![str_2](models/model_47/metrics/2023-01/day_curve_2023-01-29.png)

2. Resultats sur des prévisions dans 1h avec img cropped
### 1. Model_52
Modèle entraîné sur une vue de la dôle 3 images en séquence temporelle tout les données
- **Loss**
![str_2](models/model_52/loss_log_all.png)

- ### Modèle 52 Metrics
#### Rapport détaillé des métriques

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 88.26     | 90.79     |
| **RMSE**                      | 117.46    | 121.73    |
| **Erreur relative moyenne**    | 0.54      | 0.42      |
| **Accuracy (tolérance=20.0)** | 0.0546    |           |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 78.43
- **RMSE** : 119.47
- **Erreur relative moyenne** : 1.70

---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 95.34     | 106.09    |
| **RMSE**                      | 117.39    | 125.21    |
| **Erreur relative moyenne**    | 0.51      | 0.29      |

- **Delta Nyon-Dôle** : MAE = 127.80, RMSE = 151.96, Erreur relative = 1.15

---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 85.43     | 83.90     |
| **RMSE**                      | 101.14    | 100.63    |
| **Erreur relative moyenne**    | 0.58      | 0.46      |

- **Delta Nyon-Dôle** : MAE = 64.87, RMSE = 82.41, Erreur relative = 1.84

---
### 1. Model_51
Modèle entraîné sur une vue de la dôle 3 images en séquence temporelle avec que des images
- **Loss**
![str_2](models/model_51/loss_log_all.png)

- ### Modèle 51 Metrics
#### Rapport détaillé des métriques

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 121.58    | 110.63    |
| **RMSE**                      | 161.04    | 146.29    |
| **Erreur relative moyenne**    | 0.77      | 0.56      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 102.36
- **RMSE** : 149.64
- **Erreur relative moyenne** : 1.50

---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 131.91    | 136.86    |
| **RMSE**                      | 154.71    | 158.25    |
| **Erreur relative moyenne**    | 0.67      | 0.37      |

- **Delta Nyon-Dôle** : MAE = 150.03, RMSE = 179.75, Erreur relative = 1.20

---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 109.87    | 98.56     |
| **RMSE**                      | 129.35    | 117.71    |
| **Erreur relative moyenne**    | 0.86      | 0.65      |

- **Delta Nyon-Dôle** : MAE = 79.04, RMSE = 98.37, Erreur relative = 1.59

### 1. Model_53
Modèle entraîné sur une vue de la dôle 3 images en séquence temporelle avec tout les donnés sauf radiation
- **Loss**
![str_2](models/model_53/loss_log_all.png)
Pour ce qui concerne un modèle que avec les images --> modèle_45 meiulleur, marche mieux avec images no cropped
- ### Modèle 53 Metrics
#### Rapport détaillé des métriques

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 112.46    | 107.79    |
| **RMSE**                      | 147.06    | 143.84    |
| **Erreur relative moyenne**    | 0.69      | 0.49      |

#### Statistiques globales Delta Nyon-Dôle

- **MAE** : 78.66
- **RMSE** : 117.95


---

#### Jours de stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 100.41    | 96.93     |
| **RMSE**                      | 117.73    | 114.68    |
| **Erreur relative moyenne**    | 0.61      | 0.26      |

- **Delta Nyon-Dôle** : MAE = 126.90, RMSE = 150.31

---

#### Jours sans stratus

| Métrique                      | Nyon      | Dôle      |
|-------------------------------|-----------|-----------|
| **MAE**                       | 111.85    | 102.66    |
| **RMSE**                      | 129.06    | 118.89    |
| **Erreur relative moyenne**    | 0.74      | 0.54      |

- **Delta Nyon-Dôle** : MAE = 67.04, RMSE = 85.13
### Comparaison entre les modèles 52, 51 et 53

| Métrique                      | Model_52 (Nyon/Dôle) | Model_51 (Nyon/Dôle) | Model_53 (Nyon/Dôle) |
|-------------------------------|----------------------|----------------------|----------------------|
| **MAE**                       | 88.26 / 90.79        | 121.58 / 110.63      | 112.46 / 107.79      |
| **RMSE**                      | 117.46 / 121.73      | 161.04 / 146.29      | 147.06 / 143.84      |
| **Erreur relative moyenne**    | 0.54 / 0.42          | 0.77 / 0.56          | 0.69 / 0.49          |

#### Delta Nyon-Dôle (global)

| Métrique      | Model_52 | Model_51 | Model_53 |
|---------------|----------|----------|----------|
| **MAE**       | 78.43    | 102.36   | 78.66    |
| **RMSE**      | 119.47   | 149.64   | 117.95   |

#### Jours de stratus

| Métrique      | Model_52 (Nyon/Dôle) | Model_51 (Nyon/Dôle) | Model_53 (Nyon/Dôle) |
|---------------|----------------------|----------------------|----------------------|
| **MAE**       | 95.34 / 106.09       | 131.91 / 136.86      | 100.41 / 96.93       |
| **RMSE**      | 117.39 / 125.21      | 154.71 / 158.25      | 117.73 / 114.68      |
| **Erreur rel.** | 0.51 / 0.29        | 0.67 / 0.37          | 0.61 / 0.26          |

| Delta Nyon-Dôle | Model_52 | Model_51 | Model_53 |
|-----------------|----------|----------|----------|
| **MAE**         | 127.80   | 150.03   | 126.90   |
| **RMSE**        | 151.96   | 179.75   | 150.31   |

#### Jours sans stratus

| Métrique      | Model_52 (Nyon/Dôle) | Model_51 (Nyon/Dôle) | Model_53 (Nyon/Dôle) |
|---------------|----------------------|----------------------|----------------------|
| **MAE**       | 85.43 / 83.90        | 109.87 / 98.56       | 111.85 / 102.66      |
| **RMSE**      | 101.14 / 100.63      | 129.35 / 117.71      | 129.06 / 118.89      |
| **Erreur rel.** | 0.58 / 0.46        | 0.86 / 0.65          | 0.74 / 0.54          |

| Delta Nyon-Dôle | Model_52 | Model_51 | Model_53 |
|-----------------|----------|----------|----------|
| **MAE**         | 64.87    | 79.04    | 67.04    |
| **RMSE**        | 82.41    | 98.37    | 85.13    |

---

#### Synthèse comparative

- **Le modèle 52** (images + données météo, images crop) est le plus performant sur toutes les métriques principales (MAE, RMSE, erreur relative), aussi bien globalement que pour les jours de stratus et sans stratus.
- **Le modèle 53** (images + données météo sans radiation) est légèrement moins performant que le modèle 52, mais reste bien meilleur que le modèle 51.
- **Le modèle 51** (images seules) présente les erreurs les plus élevées, ce qui confirme l’importance d’ajouter les données météo pour améliorer la performance.
- Sur les jours sans stratus, le modèle 52 garde l’avantage, suivi du modèle 53, puis du modèle 51.
- **Conclusion** : Le modèle 52 est globalement le plus robuste et performant, le modèle 53 reste compétitif, tandis que le modèle 51 est nettement moins performant sans données météo.

## Comparaison détaillée des modèles 48 et 52

| Métrique                      | Model_48 (Nyon/Dôle) | Model_52 (Nyon/Dôle) |
|-------------------------------|----------------------|----------------------|
| **MAE**                       | 92.34 / 99.39        | 88.26 / 90.79        |
| **RMSE**                      | 121.66 / 130.79      | 117.46 / 121.73      |
| **Erreur relative moyenne**    | 0.63 / 0.57          | 0.54 / 0.42          |

### Delta Nyon-Dôle (global)

| Métrique      | Model_48 | Model_52 |
|---------------|----------|----------|
| **MAE**       | 85.42    | 78.43    |
| **RMSE**      | 130.62   | 119.47   |

### Jours de stratus

| Métrique      | Model_48 (Nyon/Dôle) | Model_52 (Nyon/Dôle) |
|---------------|----------------------|----------------------|
| **MAE**       | 72.40 / 98.91        | 95.34 / 106.09       |
| **RMSE**      | 92.71 / 119.26       | 117.39 / 125.21      |
| **Erreur rel.** | 0.37 / 0.25        | 0.51 / 0.29          |
| **Delta MAE** | 102.26               | 127.80               |
| **Delta RMSE**| 126.17               | 151.96               |

### Jours sans stratus

| Métrique      | Model_48 (Nyon/Dôle) | Model_52 (Nyon/Dôle) |
|---------------|----------------------|----------------------|
| **MAE**       | 89.98 / 91.10        | 85.43 / 83.90        |
| **RMSE**      | 107.28 / 108.92      | 101.14 / 100.63      |
| **Erreur rel.** | 0.69 / 0.60        | 0.58 / 0.46          |
| **Delta MAE** | 73.33                | 64.87                |
| **Delta RMSE**| 94.65                | 82.41                |

---

### Synthèse comparative

- **Globalement**, le modèle 52 surpasse le modèle 48 sur toutes les métriques principales (MAE, RMSE, erreur relative) pour Nyon et Dôle, ainsi que sur les jours sans stratus.
- **Sur les jours de stratus**, le modèle 48 garde l’avantage avec des erreurs plus faibles (MAE/RMSE/erreur relative) et des deltas Nyon-Dôle plus faibles.
- **Conclusion** : Le modèle 52 est plus performant globalement et sur les jours sans stratus, tandis que le modèle 48 reste meilleur pour la détection des jours de stratus. Le choix dépendra donc de l’importance relative accordée à la performance sur les jours de stratus ou sur l’ensemble des jours.

- Quelques exemples du modèle_52
![str2](models/model_52/metrics/2023-01/day_curve_2023-01-26.png)
![str2](models/model_52/metrics/2023-03/day_curve_2023-03-05.png)
![str2](models/model_52/metrics/2023-09/day_curve_2023-09-25.png)
![str2](models/model_52/metrics/2024-10/day_curve_2024-10-25.png)
![str2](models/model_52/metrics/2024-11/day_curve_2024-11-07.png)
![str2](models/model_52/metrics/2024-11/day_curve_2024-11-09.png)
![str2](models/model_52/metrics/2024-12/day_curve_2024-12-01.png)


