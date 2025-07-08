# Choix du modèle final

A partir des résultats de toutes les expériences que j'ai faites, j'ai décidé d'opter comme choix finale du modèle le fait de prendre le rayonnement de Genève et de la Dole comme output pour le train et pour l'affichage des résultats, car sur les images de la Dole on peut voir Genève et non Nyon. J'ai donc décidé d'entraîner mon modèle avec ces données, en prenant tous les mois(hiver automne)et en fixant une période d'une quizeine de jours de stratus pour comparer plus facilement les résultats. Dans les semaines précédentes, j'avais analysé les résultats et constaté que l'ajout ou la suppression de données météorologiques pouvait conduire à des améliorations dans certains cas, mais pas dans d'autres. Il est donc difficile de déterminer si le modèle fonctionne mieux sans ou avec les données météorologiques. C'est pourquoi j'ai décidé d'analyser ces deux cas de manière un peu plus approfondie sur des jours fixes afin de mieux comprendre le comportement de mon modèle.

Nous commençons par une vision globale. 

J'entraîne les deux modèles pour des prédictions à 10 min 30 min 1h et deux heures.

J'ai concentré donc mon analyse sur les jours de stratus vu que c'est la tâche qu'on cherche à resoudre

J'ai donc obtenu les erreurs suivantes.

![alt text](image.png)


Les deux premiers graphiques montrent la MAE de DOle et de Genève et le dernier montre la MAE de la différence entre les valeurs attendues et les valeurs prédites.
Comme nous pouvons le constater, nous avons un MAE qui n'augmente pas linéairement et nous n'avons pas un modèle qui soit strictement meilleur que l'autre dans tous les cas. Ce comportement a déjà été observé au cours des dernières semaines. J'ai donc essayé d'analyser plus spécifiquement le comportement des deux modèles.

Si l'on analyse le comportement des deux modèles avec des prédictions à 10 min. Je voulais essayer de tracer un scatter plot pour voir la distribution des delta des valeurs prédites par rapport aux valeurs réelles afin de voir dans quelle mesure elles s'écartent du fit parfait.
Ce qui a améné à avoir les résultats suivants: 

<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_2/metrics/delta_comparison_scatter_stratus.png" alt="Scatter plot modèle images" width="1000"/>
        </td>
        <td>
            <b>Modèle avec données météo et images</b><br>
            <img src="models/model_6/metrics/delta_comparison_scatter_stratus.png" alt="Scatter plot modèle images + météo" width="1000"/>
        </td>
    </tr>
</table>

Comme on peut le voir, la distribution des points est beaucoup plus agrégée dans le modèle avec les données météorologiques. 
Nous pouvons constater que le modèle avec tout a tendance à prédire des valeurs plus élevées que les valeurs réelles, alors que le modèle avec seulement des images a le comportement inverse.
Il serait maintenant intéressant de comprendre ce qui se passe aux dates où nous avons beaucoup de valeurs outiliers et de comparer plus en détail chaque jour.

Pour ce faire, j'utilise des cartes thermiques et je cherche à savoir à quelles heures de la journée et quels jours nous avons des deltas élevés. Une fois ces points isolés, nous pouvons analyser et comparer les courbes entre elles.
<table>
    <tr>
        <td>
            <b>Heatmap modèle avec que les images</b><br>
            <img src="models/model_2/metrics/delta_heatmap.png" alt="Heatmap modèle tous les données" width="2000"/>
        </td>
        <td>
            <b>Heatmap modèle avec img + meteo</b><br>
            <img src="models/model_6/metrics/delta_heatmap.png" alt="Heatmap modèle que images" width="2000"/>
        </td>
    </tr>
</table>

Nous constatons qu'il y a beaucoup plus de cas problématiques dans le modèle avec seulement les images. Prenons par exemple le jour 2023-03-02, 2024-10-19, 2024-11-16: 

<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_2/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_6/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_2/metrics/2024-10/day_curve_2024-10-19.png" alt="Courbe du jour 2023-10-19 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_6/metrics/2024-10/day_curve_2024-10-19.png" alt="Courbe du jour 2023-10-19 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_2/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_6/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>


Mais il y a aussi de s jours dans lesquels les deux marchent bien 2024-10-25, 2024-10-30, 2024-11-03:
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_2/metrics/2024-11/day_curve_2024-11-03.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_6/metrics/2024-11/day_curve_2024-11-03.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_2/metrics/2024-10/day_curve_2024-10-25.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_6/metrics/2024-10/day_curve_2024-10-25.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_2/metrics/2024-10/day_curve_2024-10-30.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_6/metrics/2024-10/day_curve_2024-10-30.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
On constate que dans le modèle où l'on a tout, on a presque toujours un retard de 10 minutes sur les variables prédites, alors que dans celui avec les images, on a beaucoup plus de cas où la disparition du stratus est en retard, mais dans quelques cas il disparaît sans aucun retard. Ainsi, lorsque nous disposons des données météorologiques, le modèle accorde beaucoup plus de confiance aux données météorologiques entrantes, mais lorsque nous n'en disposons pas, nous ne pouvons nous appuyer que sur les images, où parfois ce que nous voyons ne correspond pas aux capteurs. De plus, même si la valeur est en retard de 10 minutes, on peut toujours voir que le starto tend à disparaître plus lentement s'il n'atteint pas un delta entre les deux variations de presque zéro, on peut déjà supposer qu'il est en train de disparaître.

Avant de décider la version à utiliser, je veux voir si le retard est visible avec des prévisions à une heure et 30 minutes.

Concernant les prévisions à 30 minutes. 
<table>
    <tr>
        <td>
            <b>Heatmap modèle avec que les images</b><br>
            <img src="models/model_10/metrics/delta_heatmap.png" alt="Heatmap modèle avec que les images" width="2000"/>
        </td>
        <td>
            <b>Heatmap modèle avec tous les donnés</b><br>
            <img src="models/model_7/metrics/delta_heatmap.png" alt="Heatmap modèle avec tous les données" width="2000"/>
        </td>
    </tr>
</table>
On voit aussi ici que les moèle avec tous à des herreurs beaucoup plus bas.
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_10/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_7/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_10/metrics/2024-10/day_curve_2024-10-19.png" alt="Courbe du jour 2023-10-19 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_7/metrics/2024-10/day_curve_2024-10-19.png" alt="Courbe du jour 2023-10-19 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_10/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_7/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>



<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_10/metrics/2024-11/day_curve_2024-11-03.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_7/metrics/2024-11/day_curve_2024-11-03.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_10/metrics/2024-10/day_curve_2024-10-25.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_7/metrics/2024-10/day_curve_2024-10-25.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_10/metrics/2024-10/day_curve_2024-10-30.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_7/metrics/2024-10/day_curve_2024-10-30.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_10/metrics/2024-12/day_curve_2024-12-26.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_7/metrics/2024-12/day_curve_2024-12-26.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
Sur quelques examples on voit qu'on du retard en é gal manière sur les deux modèles mais un voit une tendece à copier et coller les données meteodans le modèle ou on as tous un peu plus forte. 


Après j'ai analiser les predictions dans 1 heures.

De toute façon, je regarde la distribution des données

<table>
    <tr>
        <td>
            <b>Modèle avec images </b><br>
            <img src="models/model_1/metrics/delta_comparison_scatter_stratus.png" alt="Distribution des données - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + meteo</b><br>
            <img src="models/model_0/metrics/delta_comparison_scatter_stratus.png" alt="Distribution des données - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>

Dans ce cas, la distribution des données est beaucoup plus éparpillée dans les deux cas ; examinons donc les journées plus en détail.

<table>
    <tr>
        <td>
            <b>Heatmap modèle avec que les images</b><br>
            <img src="models/model_1/metrics/delta_heatmap.png" alt="Heatmap modèle avec que les images" width="2000"/>
        </td>
        <td>
            <b>Heatmap modèle avec tous les donnés</b><br>
            <img src="models/model_0/metrics/delta_heatmap.png" alt="Heatmap modèle avec tous les données" width="2000"/>
        </td>
    </tr>
</table>


Nous constatons que le modèles avec des images fait mois d'herreur grave on peut donc comparer les curves pour voir les resultats. 

<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_1/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_0/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_1/metrics/2024-10/day_curve_2024-10-19.png" alt="Courbe du jour 2023-10-19 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_0/metrics/2024-10/day_curve_2024-10-19.png" alt="Courbe du jour 2023-10-19 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_1/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_0/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>


<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_1/metrics/2024-11/day_curve_2024-11-03.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_0/metrics/2024-11/day_curve_2024-11-03.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_1/metrics/2024-10/day_curve_2024-10-25.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_0/metrics/2024-10/day_curve_2024-10-25.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images</b><br>
            <img src="models/model_1/metrics/2024-10/day_curve_2024-10-30.png" alt="Courbe du jour 2024-11-16 - Modèle images" width="2000"/>
        </td>
        <td>
            <b>Modèle avec images + données météo</b><br>
            <img src="models/model_0/metrics/2024-10/day_curve_2024-10-30.png" alt="Courbe du jour 2024-11-16 - Modèle images + météo" width="2000"/>
        </td>
    </tr>
</table>

Le modèle basé sur l'image seule présente moins de décalage dans l'ensemble et la disparition du stratus se fait progressivement.

Le modèle avec images est donc celui qui permet de s'assurer que l'on a vraiment des prévisions et non des copier-coller, même si dans certains cas on a des valeurs plus élevées en termes d'erreur.

Je voulais aussi faire un test pour savoir si un modèle à sorties multiples pouvait aider à obtenir de meilleurs résultats. J'ai donc essayé un modèle avec et sans images à sorties multiples. Les voici comparés en termes d'erreur

![alt text](mae_delta_combined_all.png)

Vu l'mae bas j'ai volu regarder si aussi dans les modèles avec tous les données à output mulitple on voyait le decalage.
En effet ce que je peux observer globalement  est un decalge mineur mais toujours présents sur les prediction à 10 minutes surtout. 
Si on compare un resultats. 


<table>
    <tr>
        <td>
            <b>Modèle avec output multiples avec tout</b><br>
            <img src="models/model_11/metrics/2023-03/prediction_curves_2023-03-02_test.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output que images prediction dans une heure</b><br>
            <img src="models/model_1/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output images + meteo prediction dans une heure</b><br>
            <img src="models/model_0/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
          <td>
        <b>Modèle avec single output que images prediction dans 10 minutes</b><br>
            <img src="models/model_2/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output images + meteo prediction dans 10 minutes</b><br>
            <img src="models/model_6/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
    </tr>
</table>


<table>
    <tr>
        <td>
            <b>Modèle avec output multiple avce tout</b><br>
            <img src="models/model_11/metrics/2024-11/prediction_curves_2024-11-16_test.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output que images prediction dans une heure</b><br>
            <img src="models/model_1/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output images + meteo prediction dans une heure</b><br>
            <img src="models/model_0/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
          <td>
        <b>Modèle avec single output que images prediction dans 10 minutes</b><br>
            <img src="models/model_2/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output images + meteo prediction dans 10 minutes</b><br>
            <img src="models/model_6/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
    </tr>
</table>

Nous pouvons constater que le modèle à sorties multiples tend à être beaucoup moins précis pour les prévisions à 10 minutes que le modèle à sortie unique avec toutes les données, mais à l'heure, il est meilleur, mais pas aussi bon que les modèles à sortie unique avec seulement des images.
Il était donc intéressant d'examiner le modèle de prédiction multiple avec les seules images pour voir s'il pouvait donner de meilleurs résultats. 
J'ai regarder les previsions à 60 minutes avec output multiples.
J'ai examiné la dispersion des données.
<table>
    <tr>
        <td>
            <img src="models/model_3/metrics/delta_comparison_scatter_stratus.png" alt="Scatter plot modèle à sorties multiples" width="2000"/>
        </td>
        <td>
            <img src="models/model_1/metrics/delta_comparison_scatter_stratus.png" alt="Scatter plot modèle à sortie unique" width="2000"/>
        </td>
    </tr>
    <tr>
        <td align="center"><b>Modèle à sorties multiples que imagess</b></td>
        <td align="center"><b>Modèle à sortie unique que images</b></td>
    </tr>
</table>

Les deux données étant très dispersées, je consulte les heatmap pour obtenir plus d'informations.

<table>
    <tr>
        <td>
            <img src="models/model_3/metrics/delta_heatmap_t5.png" alt="Heatmap modèle à sorties multiples" width="2000"/>
        </td>
        <td>
            <img src="models/model_1/metrics/delta_heatmap.png" alt="Heatmap à sortie unique" width="2000"/>
        </td>
    </tr>
    <tr>
        <td align="center"><b>Modèle à sorties multiples</b></td>
        <td align="center"><b>Modèle à sortie unique</b></td>
    </tr>
</table>




Je constate que les erreurs sont nettement plus faibles dans le modèle à sortie unique. Voici quelques comparaisons

<table>
    <tr>
        <td>
            <b>Modèle avec output multiples que images</b><br>
            <img src="models/model_3/metrics/2023-03/prediction_curves_2023-03-02_test.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output que images prediction dans une heure</b><br>
            <img src="models/model_1/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
          <td>
            <b>Modèle avec single output que images prediction dans 10 minutes</b><br>
            <img src="models/model_2/metrics/2023-03/day_curve_2023-03-02.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
    </tr>
</table>

<table>
    <tr>
        <td>
            <b>Modèle avec output multiples que images</b><br>
            <img src="models/model_3/metrics/2024-11/prediction_curves_2024-11-03_test.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output que images prediction dans une heure</b><br>
            <img src="models/model_1/metrics/2024-11/day_curve_2024-11-03.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
        <td>
            <b>Modèle avec single output que images prediction dans 10 minutes</b><br>
            <img src="models/model_2/metrics/2024-11/day_curve_2024-11-03.png" alt="Courbe du jour 2023-03-02" width="2000"/>
        </td>
    </tr>
</table>


En général, le modèle a tendance à se dégrader tant pour les prévisions de l'heure suivante que pour celles des 10 prochaines minutes.

Les modèles pour lesquels j'étais indécis sont donc les 3 suivants.


| Modèle    | Type de données           | Sortie            | Δ MAE Moyen (W/m²) | Description |
|-----------|--------------------------|-------------------|--------------------|-------------|
| Modèle 1  | Images + données météo   | Sortie unique     | 75,75              | Le modèle a tendance à se tromper moins souvent sur la prédiction de la disparition de la couche, mais il y a toujours un retard, particulièrement visible pour l’horizon de 10 minutes. Ce retard reste acceptable car le modèle détecte tout de même la disparition du stratus. Cependant, pour des prévisions à plus long terme, ce phénomène de "copier-coller" des données météo dégrade les performances, et le modèle perd en capacité de vraie prévision. |
| Modèle 2  | Images + données météo   | Sorties multiples | 75,13              | Ce modèle parvient à être meilleur en termes de décalage que le modèle précédent, bien que les prévisions à 10 minutes soient plus incorrectes que le modèle à sortie unique, mais tendent à être meilleures à l'heure.|
| Modèle 3  | Images uniquement        | Sortie unique     | 80,5               | Le modèle basé uniquement sur les images ne détecte pas toujours la disparition de la couche au bon moment, mais il garantit que nous n'avons pas de données d'entrée copiées-collées et il est dans certains cas meilleur que le modèle à sorties multiples, comme le montrent les comparaisons ci-dessus. Ainsi, même si nous avons un delta mae légèrement plus important, il est toujours préférable d'utiliser uniquement les images pour prédire et garantir la prédiction. En outre, nous pouvons également constater que, dans certains cas, la disparition de la couche n'est pas vraiment décalée et que, par conséquent, ce que nous voyons sur les images ne correspond pas au modèle et qu'il est donc plus difficile de trouver une corrélation entre les images et le modèle (un exemple ci-dessous). Ceci montre donc que l'on accorde beaucoup d'importance aux données météorologiques, qui ne sont parfois pas représentatives lors de l'utilisation des données météorologiques.|


Exemple de non-corrélation entre les images et les données météorologiques de rayonnement 2024-11-16

<table>
    <tr>
        <td>
            <b>Modèle avec seulement les images (10 min)</b><br>
            <img src="models/model_2/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images 10 min" width="2000"/>
        </td>
        <td>
            <b>Modèle avec seulement les images (30 min)</b><br>
            <img src="models/model_10/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images 30 min" width="2000"/>
        </td>
        <td>
            <b>Modèle avec seulement les images (1h)</b><br>
            <img src="models/model_1/metrics/2024-11/day_curve_2024-11-16.png" alt="Courbe du jour 2024-11-16 - Modèle images 1h" width="2000"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <th>09:50</th>
        <th>11:10</th>
        <th>11:20</th>
        <th>11:40</th>
        <th>12:00</th>
        <th>12:10</th>
        <th>12:20</th>
        <th>12:50</th>
        <th>13:20</th>
    </tr>
    <tr>
        <td><img src="analysis/1159_2_2024-11-16_0950.jpeg" alt="09:50" width="480"/></td>
        <td><img src="analysis/1159_2_2024-11-16_1110.jpeg" alt="11:10" width="480"/></td>
        <td><img src="analysis/1159_2_2024-11-16_1120.jpeg" alt="11:20" width="480"/></td>
        <td><img src="analysis/1159_2_2024-11-16_1140.jpeg" alt="11:40" width="480"/></td>
        <td><img src="analysis/1159_2_2024-11-16_1200.jpeg" alt="12:00" width="480"/></td>
        <td><img src="analysis/1159_2_2024-11-16_1210.jpeg" alt="12:10" width="480"/></td>
        <td><img src="analysis/1159_2_2024-11-16_1220.jpeg" alt="12:20" width="480"/></td>
        <td><img src="analysis/1159_2_2024-11-16_1250.jpeg" alt="12:50" width="480"/></td>
        <td><img src="analysis/1159_2_2024-11-16_1320.jpeg" alt="13:20" width="480"/></td>
    </tr>
</table>

# TODO
Changer le mots outliers

Changer la granularite des heatmap 

Combiner les pred sur les graphes

Compredre pourquoi avec que les images ça marche mieux que avec tous les données à 1h 