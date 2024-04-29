# perceptron

En 1957, Frank Rosenblatt crée la premier algorithme d'apprentissage supervisé, le perceptron.

## But du projet

L'objectif est de comprendre la base du fonctionnement des réseaux de neurones en recréant le premier model mathématique du neurone avec la loi de Widrow-Hoff.
Pour aller plus loin j'ai même reproduit l'expérience fait par Frank sur la classification des Iris.

## Contenu
### Main file
**iris.py :** contient le model, charge le dataset iris.csv et prédit le résultat de classification
**exam.py :** même exercice que iris.py mais sur d'autres données

### Class
**/akiplot/akiplot.py :** une surclasse crée pour faciliter la création de graphique avec MathPlotLib.
**/pitch/perceptron.py :** le perceptron en lui même.
**/pitch/vprint.py :**  une surclasse à print() permettant de faciliter l'affichage en mode verbose du model
**/pitch/pitch_class.py :** class de création du model incluant : préparation des données d'entrainement et de prédiction, création des graphs, entrainement, prédiction, verbose, etc.
