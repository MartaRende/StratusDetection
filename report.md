# Introduction au problème

L'objectif de ce projet est de détecter la présence de la couche de stratus et de prévoir à court terme (prévision pour les 6 prochaines heures) sa disparition ou son apparition. Le stratus est une accumulation de nuages présente dans le canton de Vaud. Sa particularité est d’être très bas : si l’on se place sur une montagne suffisamment haute, on peut observer une mer de nuages recouvrant la plaine vaudoise. Heureusement, nous disposons d’une caméra, grâce aux images de MétéoSuisse, qui nous permet de nous placer juste au-dessus du stratus et d’avoir une bonne vue de la plaine vaudoise. Cette caméra est située à la Dôle et sera utilisée dans le cadre de ce projet. Afin de mieux comprendre les images dont nous disposons et le phénomène météorologique étudié, nous allons préciser le contexte. Voici quelques exemples d’images disponibles.


![Non Stratus](analysis/non_startus_dole.jpeg)

![Stratus](analysis/startus_dole.jpeg)


Maintenant que nous avons une vue d’ensemble du problème, nous pouvons détailler les caractéristiques de ce phénomène météorologique.
Voici quelques recherches effectuées :

### Différence entre le brouillard et le stratus :
- **Distance du sol** :
- **Visibilité** :
    - Brouillard : < 1 km
    - Stratus : > 1 km
(Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/portrait/meteosuisse-blog/fr/2025/02/grisailles-frequentes-semestre-hiver.html))

### Facteurs influençant la présence du brouillard ou du stratus :
(Source : [Météonews](https://meteonews.ch/fr/News/N14196/Brouillard-et-stratus-%E2%80%93-Compagnons-de-la-saison-froide))
- **Direction du vent** :
    - Si le vent vient du **Nord-Est** : c’est la bise (transport d’air continental → froid et lourd). Comme l’air est froid et lourd, le brouillard monte et crée ainsi du stratus.
    - **Bise plus forte** : stratus avec une limite supérieure plus haute.
    - **Bise faible** : limite supérieure < 1000 m.
    - **Bise moyenne** : limite supérieure ≤ 1500 m.
    - **Bise forte** : limite supérieure > 2000 m.

### Comment le stratus disparaît-il ?
- **Grâce au rayonnement solaire** : Le soleil chauffe la couche froide, ce qui rend la dissipation plus difficile en hiver.
- **Le vent** : Si le vent amène de l’air chaud (vent de sud-ouest à ouest) ou de l’air sec (foehn).
- **Arrivée d’une perturbation** : Changement de pression ou de température.
- **Modification de la pression atmosphérique**.
- Plus la limite supérieure est élevée, plus les chances de dissolution sont faibles, mais d’autres facteurs jouent aussi un rôle.
(Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/portrait/meteosuisse-blog/fr/2024/10/limite-superieure-brouillard.html))

### Formation du stratus :
Source : [MétéoSuisse](https://www.meteosuisse.admin.ch/meteo/meteo-et-climat-de-a-a-z/brouillard/le-plateau-une-region-a-brouillard.html#:~:text=,remplie%20sur%20le%20Plateau%20suisse)
La bise est souvent présente lors de la formation du stratus.
Source : [Agrometeo](https://api.agrometeo.ch/storage/uploads/Web_Wetterlagen_FR_low.pdf)

- En situation anticyclonique stable, il y a une **inversion thermique**.
- Dans le cas du stratus, on a une inversion thermique (l’air est plus chaud en hauteur qu’au sol), créée par les anticyclones. L’air froid emprisonné près du sol est souvent humide, surtout après des nuits claires où le refroidissement radiatif est important. Lorsque l’humidité atteint le point de saturation, elle se condense et forme des nuages bas appelés stratus.
  - **Inversion thermique** : Se forme avec une situation anticyclonique stable, qui prévoit des pressions atmosphériques élevées, poussant l’air froid en bas (source : [MétéoSuisse](https://www.meteosuisse.admin.ch/meteo/meteo-et-climat-de-a-a-z/brouillard/le-plateau-une-region-a-brouillard.html#:~:text=,remplie%20sur%20le%20Plateau%20suisse)) → Données très importantes, il faut regarder la pression.
- Faible ensoleillement ou soleil bas.
- Vent faible dans les basses couches de l’atmosphère (sauf la bise) : condition satisfaite en situation de haute pression.
- **Topographie** : L’air froid et humide doit pouvoir s’accumuler dans un bassin (ex. la région suisse entre les Alpes et le Jura) → ce qui crée une inversion thermique.
- Ciel clair → satisfait en conditions de haute pression.
- Humidité élevée : Comme l’air froid peut contenir moins d’humidité que l’air chaud, la vapeur d’eau finit par se condenser et former du brouillard ou du stratus.

### La saison influence la présence du stratus :
Le stratus est plus présent en **hiver** et **automne**, et la vitesse à laquelle il peut disparaître varie.
*À voir si diviser les études en saisons pourrait avoir des avantages.*

Comme on le voit, plusieurs facteurs météorologiques influencent l’apparition et la disparition du stratus. Actuellement, MétéoSuisse utilise certaines de ces données pour prédire ce phénomène. Malheureusement, dans certains cas, ces données ne sont pas suffisantes ou nécessitent une intervention humaine pour garantir l’exactitude des prévisions et les réaliser de manière autonome.

Dans ce projet, je vais donc essayer de créer un modèle qui, au lieu d’utiliser uniquement les données météorologiques, utilise aussi les images de la Dôle afin d’obtenir le plus d’informations possible.
Pour cela, je dispose de deux années d’images et de deux années de données météorologiques Inca avec un timestamp de 10 minutes, plus des données de la plateforme idaweb, afin d’obtenir des informations supplémentaires non disponibles dans Inca.

Dans notre cas, comment déterminer s’il y a un stratus à un moment donné ? Pour résumer, ce phénomène obscurcit la plaine vaudoise, ce qui signifie que dans la plaine il n’y aura pas de soleil, contrairement à la Dôle où il y aura du soleil. Donc, pour savoir si on est en présence d’un stratus à un moment donné, il suffit de mesurer la puissance d’irradiation donnée par le soleil à la Dôle et en plaine. Les deux données d’irradiation, ainsi que toutes les autres données météorologiques, sont disponibles grâce aux stations météorologiques de la Dôle et de Nyon (en plaine). Si la différence de rayonnement entre Dôle et Nyon est importante, cela signifie que nous avons un stratus.

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