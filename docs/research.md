# Recherche

## Doc sur le startus
## Example de mesure du rayonnement solaire


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
    *TODO : Chercher ce qu'est une bise forte, moyenne ou faible.*


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

**Formation du stratus** :
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

---

## Architecture du modèle possible

### Phase 1
- Extraction des **features** d'une image avec un **CNN** qui pourrait aider à la classification binaire.
- Utilisation d'un **NN** (ex. MLP) pour traiter les **features** des données INCA.
- Cela permettra de déterminer s'il y a du stratus sur une image.

### Pour la prédiction du stratus :
- Utilisation d'un **RNN** (Recurrent Neural Network) :  
  Source : [GeeksforGeeks](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)
  - **Architecture many-to-one** utilisée pour traiter une suite d'images dans le temps.
  - Peut aussi être utilisé pour les données INCA.
  
Ou bien, combiner un **MLP** qui prédit la présence du stratus avec les données INCA + un **RNN (LSTM)** entraîné sur les images du passé.
---
### PLANNING initiale
---
[git](https://github.com/users/MartaRende/projects/2/views/1)

### Format d'image
  - 26729x3872

### CNN plus detaillé
[Description de la difference entre chaaque layer](https://medium.com/@RobuRishabh/convolutional-neural-network-cnn-part-1-d1c027913b2b)
1. Pour les premiers test commencer avec des covolutions --> utile pour trouver des features/patterns important dans l'image 
2. Pour la fonctins d'activation --> commencer avec une [ReLu](https://medium.com/@sourenh94/understanding-relu-activation-function-in-convolutional-neural-networks-691614493bcb) --> introduction de la non linearité dans le système 
3. Appliquer une Pooling layer pour reduire la dimensionalité ou MaxPolling ou AveragePooling à tester 
4. Repeter les passage 
5. A' la fin peut etre aplliquer un dropout pour reduire l'overfitting

6. Considerer l'indroduction du [padding e du stride](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html) --> dans les premiers layer le stride doit rester à 1 pour ne pas perdre trop d'infos tout de suite le padding depand de la dimension du kernel 
### MLP for weather data
Le but c'est de rentre avec des données meteorologique et de 
1. Des layers Dense (layer fully-connected)
2. Des ReLu comme fonction d'activation
(Des dropout?)
3. TRouver un bon numero des features à mettre en sorti
 
### MLP for join CNN with the first MLP

1. Dense layers, RELu
2. Output layer Sigmoid
3. Prévoir un reseau qui a plus que un output pour faire la prevision sur les prochaines 6h

### Data Required from inca data

- RR --> precipitation (mm/ h)
- TD -->  temperature at which air becomes fully saturated with moisture causing condense (degre)
- PT --> precipitation type (valeur de 0 à 5)
- WG --> coup de vent(m/s)
- TT --> temp à 2 mètre du sol(degres)
- TW --> ?
- CT --> couverture du ciel (%)
- FF --> wind speed(m/s)
- RS --> snowfall(mm/h)
- TG --> Temperature du surface du sol (deg)
- Z0 --> hauteur à la quelle il y le zero term.
- ZS --> hauteur à laquelle il y a la neige
- ZM --> ? 
- SU --> duré relative du soleil (%)
- DD --> direction du vent en dégré

## Questions

1. **Accès à Chacha**.
2. **Temps écoulé entre deux images**.
3. **Données qui peuvent être utiles sans les images** : direction et puissance des vents, pression, humidité, température, données d'ensoleillement, (couvereture du ciel).
4. **L'output du modèle doit-il être binaire ou devons-nous être en mesure de dire si le stratus est en train de disparaître ou s'il est encore là ?**
5. **Plannification ?**
6. **Donnes inca predit dans le futur?**

