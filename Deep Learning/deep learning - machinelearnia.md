# Bases du Deep Learning

### Définitions de base

> Domaine du [[machine learning]] où l'on développe des *réseaux de neurones artificiels*.

Le fonctionnement est le même qu'en ML mais le modèle change : on a un réseau de fonctions connectées les unes aux autres -> un *réseau de neurones*.

Plus le réseau est profond, plus les tâches réalisables peuvent être complexes : on parle donc d'**apprentissage profond**.

---
### Point historique

**1943** : Premiers réseaux de neurones par *Warren McCulloch* et *Walter Pitts*.
Ils s'inspirent notamment des neurones biologiques (cellules excitables interconnectées transmettant les signaux dans le corps) :
Réception de signaux activateurs et inhibiteurs au niveau des synapses -> à partir d'un certain seuil, activation et transmission d'un signal électrique vers les terminaisons -> signal reçu par d'autres neurones.
En IA, un neurone est une fonction de transfert recevant des entrées $x$ et renvoyant une sortie $y$.
Deux étapes dans ces fonctions :
- Agrégation :
$$f = w_1x_1 + w_2x_2 + w_3x_3$$
Les $w$ sont des poids (+1 ou -1) selon le caractère activateur ou inhibiteur de l'entrée.
- Activation :
$$\begin{cases} y = 1 &\text{si $f \geq 0$} \\ y = 0 &\text{sinon} \end{cases}$$
Ces premiers réseaux de neurones artificiels seront nommés *Threshold Logic Unit* (que des entrées logiques) -> permettent de reproduire certaines fonctions logiques (AND, OR).
Il a été démontré qu'en connectant plusieurs de ces fonctions les unes aux autres, on pourrait résoudre n'importe quel problème de logique booléenne $\Rightarrow$ engouement démesuré.

**Problème** : plusieurs limitations comme l'absence d'algorithmes d'apprentissage (les $w$ sont à trouver nous même).

**1957** : Premier algorithme d'apprentissage par Franck Rosenblatt, le **Perceptron**.
C'est un neurone artificiel comme précedemment mais doté à présent d'un algorithme d'apprentissage permettant de trouver les $w$ afin d'obtenir les sorties recherchées.
Inspiré de la *théorie de Hebb* : lorsque deux neurones bio sont excités conjointement, il renforcent leur lien synaptique (plasticité synaptique).
Ainsi, Rosenblatt entraîne avec des données de référence son neurone de manière à ce qu'il renforce ses paramètres W à chaque fois qu'une entrée est activée en même temps que la sortie dans les données : 
$$W = W + \alpha(y_{true} - y)X$$
Le poids correspondant à la bonne entrée est renforcé jusqu'à activation $\Rightarrow$ engouement démesuré.

**Problème** : le Perceptron est un modèle linéaire (la fonction d'aggrégation se reprèsente à l'aide d'une droite) et une grande partie des phènomènes réels n'est pas linéaire. $\Rightarrow$ $1^{er}$ hiver de l'IA

**1986** : **Perceptron multicouche** par Geoffrey Hinton. Pour résoudre le problème de linéarité, on relie plusieurs Perceptrons $\Rightarrow$ modèle non linéaire donc plus intéressant. On a plusieurs couches de neurones et un certain nombre de neurones dans chaque couche.
Mais comment bien choisir les $w$ et le biais $b$ pour avoir un bon modèle $\Rightarrow$ **Back-Propagation**.

La Back-Propagation consiste à déterminer comment la sortie du réseau varie en fonction des paramètres dans chaque couche => calcul d'une *chaîne de gradients*. On part de la fin et on calcule des dérivées partielles successives (des gradients). On peut ensuite mettre à jour les paramètres de manière à minimiser l'erreur avec la **Descente de Gradient**.

Finalement, dans l'ordre :
- **Forward Propagation** : circulation des données afin de produire une sortie
- **Cost Function** : calcul de l'erreur à l'aide d'une fonction coût
- **Backward Propagation** : mesure des variations de la fonction coût par rapport à chaque couche
- **Gradient Descent** : correction des paramètres 

Ce modèle évolua (apparition de nouvelles fonctions d'activation).

**1990** : premières variantes du Perceptron multicouche comme les **Réseaux de Neurones Convolutifs** (Yann LeCun) permettant de reconnaître et traiter des images.

**1997** : premiers Réseaux de Neurones Récurrents (lecture de texte, reconnaissance vocale)

Cependant, entraîner nécessite énormément de données $\Rightarrow$ impossible dans les années 90.
Aussi une limitation de puissance de calcul.

**2012** : compétition ImageNet, reconnaissance d'image avec une performance jamais vue (Geoffrey Hinton).

---
### Le Perceptron

> Unité de base des réseaux de neurones. Modèle de **classification binaire** capable de séparer linéairement deux classes de points.

##### Neurone et frontière de décision

La droite de séparation des points s'appelle la **frontière de décision** $\Rightarrow$ on cherche son équation.

On utilise donc un *neurone* avec entrées $X$, poids $W$ et biais $b$ (par exemple pour 2 variables) :
$$z(x_1,x_2) = w_1x_1 + w_2x_2 + b$$
La frontière de décision est d'équation $z(x_1,x_2) = 0$. On ajuste les paramètres de manière à ce qu'elle sépare au mieux les classes de points. Ainsi, les prédictions se font comme suit :
$$\begin{cases} y_{pred} = 0 &\text{si $z < 0$} \\ y_{pred} = 1 &\text{si $z \geq 0$} \end{cases}$$
###### Fonction d'activation

Pour améliorer le modèle, on peut introduire des *probabilités* : fonction d'activation renvoyant une sortie qui s'approche de 0 ou 1 à mesure qu'on s'éloigne de la frontière de décision dans un sens ou l'autre $\Rightarrow$ fonction **Sigmoïde/Logistique** : $a(z) = \frac{1}{1+e^{-z}}$

```functionplot
---
title: Sigmoïde
xLabel: z
yLabel: a(z)
bounds: [-10,10,-0.5,1.5]
disableZoom: true
grid: false
---
a(x) = 1/(1 + exp(-x))
```

Permet de *convertir la sortie* $z$ en probabilité $a(z)$ d'appartenir à la classe 1. 
Ces probabilités suivent une **loi de Bernoulli** :
$$P(Y=y) = a(z)^y \times (1 - a(z))^{1-y}$$
Pour chaque sortie, la probabilité de succés (classe 1) est donnée par $a(z)$.

Dans le neurone on a alors la sortie suivie de la fonction d'activation. À présent, on veut les paramètres tels que $a(z)$ commète le moins d'erreurs possible $\Rightarrow$ on définit une **fonction coût**.

##### Fonction coût

> Loss Function : permet de quantifier les **erreurs effectuées** par un modèle.

On peut utiliser par exemple la fonction de **Log Loss** :
$$L = -\frac{1}{m}\sum_{i=1}^{m} y_i log(a_i) + (1-y_i)log(1-a_i) 
 \qquad \begin{cases} m : \text{nombre de donn\'ees} \\ y_i : \text{donn\'ee i} \\ a_i : \text{sortie i} \end{cases}$$
***Intuition mathématique*** :

Une façon d'évaluer la pertinence est de calculer la **Vraisemblance** : indique la *plausabilité* du modèle vis-à-vis de *vraies* données.
On connaît certaines données et on va vérifier si les prédictions sont bonnes.
Si une information connue comme vraie est indiquée vraie à 80% par le modèle $\Rightarrow$ le modèle est vraisemblable à 80%.
Dans notre cas, on effectue le produit de toutes ces probabilités en remplaçant à l'aide de Bernoulli et on obtient la vraisemblance (**L**ikelihood) :
$$\mathcal{L} = \prod_{i = 1}^m a_i^{y_i} \times (1 - a_i)^{1-y_i}$$
Plus le résultat est proche de 100%, plus le modèle est vraisemblable.
**Problème** : on tombe sur des résultats proches de 0 (car produit de probabilités -> converge vers 0 lorsqu'on ajoute des facteurs) donc problème de représentation de flottants à terme $\Rightarrow$ fonction *logarithme*.
Le logarithme est une fonction monotone croissante donc conserve l'ordre des termes $\Rightarrow$ n'affecte pas la recherche de l'antécédent du max. On passe donc au logarithme :
$$\begin{split} \log(L) & = \log\bigg(\prod_{i = 1}^m a_i^{y_i} \times (1 - a_i)^{1-y_i}\bigg)\\ & = \sum_{i = 1}^m \log(a_i^{y_i} \times (1 - a_i)^{1-y_i}) \\ & = \sum_{i = 1}^m \log(a_i^{y_i}) + \log((1 - a_i)^{1-y_i}) \\ & = \sum_{i=1}^{m} y_i \log(a_i) + (1-y_i)\log(1-a_i)\end{split}$$
On trouve une expression proche du Log Loss. Comme il n'existe pas d'algorithme de maximisation, on cherche à *minimiser l'opposée* et on normalise en divisant par $m$ :
$$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m} y_i log(a_i) + (1-y_i)log(1-a_i)$$

##### Descente de Gradient

> Algorithme d'apprentissage consistant à ajuster les paramètres de façon à minimiser la fonction coût.

On doit calculer le *gradient (dérivée)* de la fonction coût par rapport aux paramètres.
La **formule de récurrence** est la suivante :
$$W_{t+1} = W_{t} - \alpha\frac{\partial \mathcal{L}}{\partial W_t} \ \ \ \ \ \ \begin{cases} W_{t} : \text{Paramètre $W$ à l'instant $t$} \\ \alpha : \text{Pas d'apprentissage positif} \\ \frac{\partial \mathcal{L}}{\partial W_t} : \text{Gradient à l'instant $t$} \end{cases}$$
- Gradient négatif : $W$ augmente
- Gradient positif : $W$ diminue

En itérant de cette manière, on se rapproche progressivement vers le minimum en descendant la courbe : *descente de gradient*.

**Condition** : il faut que la fonction soit *convexe* (qu'un seul minimum donc pas de minimum local).

Il faut également l'expression des gradients. Avec les fonctions Log Loss et $z$, cela donne :

$$\begin{cases} \frac{\partial \mathcal{L}}{\partial w_1} = \frac{1}{m}\sum_{i=1}^{m}(a_i - y_i)x_{i1}  \\ \frac{\partial \mathcal{L}}{\partial w_2} = \frac{1}{m}\sum_{i=1}^{m}(a_i - y_i)x_{i2}  \\ \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m}\sum_{i=1}^{m}(a_i - y_i)\end{cases}$$

---
### Vectorisation des équations

> Consiste à mettre les données dans un vecteur ou une matrice de façon à effectuer des opérations sur l'ensemble de ces données.

Permet de **réduire le temps de calcul** : indispensable en Deep Learning et Machine Learning.
On utilise donc du calcul matriciel et de l'algèbre linéaire.

On part d'un dataset qu'on vectorise afin d'obtenir une matrice $X \in \mathcal{M}_{m ,n} (\mathbb{R})$  et un vecteur $y \in \mathbb{R}^{m}$ avec $m$ le *nombre de données* et $n$ le *nombre de variables* du dataset.

##### Vecteur Z

On crée d'abord un vecteur $Z \in \mathbb{R}^{m}$ avec la valeur renvoyée par la fonction $z$ pour chaque donnée. On l'obtient avec l'opération :
$$Z = X \cdot W + b$$
Ici, $W \in \mathbb{R}^{n}$ est le vecteur contenant les poids $w$ et $b \in \mathbb{R}^{m}$ le vecteur contenant le biais (peut être juste un réel c.f broadcasting).

#### Vecteur A

Celui-ci est obtenu en *composant* le vecteur $Z$ par la *fonction d'activation* :
$$A = \frac{1}{1 + e^{-Z}}$$
Ici c'est une composition terme-à-terme, pas une exponentielle de matrice.

#### Vectorisation de la fonction coût

On cherche à comparer directement le vecteur $A$ au vecteur $y$ à l'aide de la fonction coût.
On insère directement les vecteurs dans l'expression de la fonction (*opérations terme-à-terme* toujours). On obtient un vecteur d'expressions qu'on *somme entre elles* (opérateur $\sum$) et on obtient un **réel** (le coût) :
$$\mathcal{L} = -\frac{1}{m}\sum y \times log(A) + (1-y) \times log(1-A)$$

##### Vectorisation de la Descente de Gradient

On utilise le vecteur $W$ et le *Jacobien* de $\mathcal{L}$ par rapport à $W$ :
$$W_{t+1} = W_{t} - \alpha \frac{\partial \mathcal{L}}{\partial W}$$
Pour le paramètre $b$, inutile de vectoriser (considéré comme un réel).

##### Vectorisation des Gradients

On utilise l'expression des gradients et on remarque que le Jacobien vérifie la relation :
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{m}X^T \cdot (A - y)$$
Pour la dérivée par rapport à $b$, on injecte dans l'expression les vecteurs $A$ et $y$ et on applique l'opérateur $\sum$ sur le vecteur obtenu $\Rightarrow$ on obtient un réel qu'on divise par $m$ :
$$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum (A - y)$$

#### Récapitulatif des équations

**Modèle** :
$$\boxed{\begin{split} Z = X \cdot W + b \\ A = \frac{1}{1 + e^{-Z}} \end{split}}$$
**Fonction Coût** :
$$\boxed{\mathcal{L} = -\frac{1}{m}\sum y \times log(A) + (1-y) \times log(1-A)}$$

**Descente de Gradient** :
$$\boxed{\begin{cases} W_{t+1} = W_{t} - \alpha \frac{\partial \mathcal{L}}{\partial W} & \bigg(\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{m}X^T \cdot (A - y)\bigg)\\ b = b - \alpha \frac{\partial \mathcal{L}}{\partial b} 
 & \bigg(\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum (A - y)\bigg)\end{cases}}$$

---
### Implémentation d'un neurone artificiel

##### Diagramme fonctionnel

Tout commence avec notre *dataset d'entraînement* de la forme $(X, \ y)$.
Le code se décompose de la manière suivante :
- **Fonction d'initialisation** ($X$) : Initialise les paramètres $W$ et $b$ du modèle (bonne taille pour $W$ notamment)
- ***Algorithme itératif*** :
	- **Fonction représentant le modèle de neurone** ($X,\  W,\  b$)  : passe les arguments aux fonctions du modèle (fonction linéaire et fonction d'activation)
	- **Fonction d'évaluation** ($A,\ y$) :  effectue le calcul de coût.
	- **Gradients** ($X,\ A,\ y$) : calcul des gradients de la fonction coût.
	- **Mise à jour** ($W,\ b,\ dW,\ dB$): met à jour les paramètres du modèle avec les gradients.

C'est la ***Descente de Gradient*** : permet de *minimiser* le coût du modèle.

![[diagramme_fonc.png]]

##### Implémentation des fonctions

Importations 

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
```

Génération d'un dataset

```python
X, y = make_blobs(n_samples=100, n_features=2, centers = 2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
plt.show()
```

Fonction d'initialisation

```python
def initialisation(X):
	W = np.random.randn(X.shape[1], 1)
	b = np.random.randn(1)
	return (W,b)
```

Fonction du modèle

```python
def model(X, W, b):
	Z = X.dot(W) + b
	A = 1 / (1 + np.exp(-Z))
	return A
```

Fonction coût

```python
def log_loss(A, y):
	return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
```

Fonction des gradients

```python
def gradients(A, X, y):
	dW = 1 / len(y) * np.dot(X.T, A - y)
	db = 1 / len(y) * np.sum(A - y)
	return (dW, db)
```

Fonction de mise à jour

```python
def update(dW, db, W, b, learning_rate):
	W = W -learning_rate * dW
	b = b - learning_rate * db
	return (W, b)
```

##### Algorithme complet

```python
def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 100):
	# Initialisation W, b
	W, b = initialisation(X)
	# liste de coûts (permet de visualiser)
	Loss = []
	# boucle d'apprentissage
	for i in range(n_iter):
		A = model(X, W, b)
		Loss.append(log_loss(A, y))
		dW, db = gradients(A, X, y)
		W, b = update(dW, db, W, b, learning_rate)
	# calcul des prédictions pour les données entraîn.
	y_pred = predict(X, W, b)
	# affichage de la précision
	print(accuracy_score(y, y_pred))
	plt.plot(Loss)
	plt.show()
	return (W, b)
```

##### Prédictions

À présent, notre modèle ($W$ et $b$ enregistrés) peut être utilisé pour réaliser des prédictions sur la classe d'autres données qu'on lui fournit. Il nous donnera une probabilité qu'on pourra interpréter.

```python
def predict(X, W, b):
	A = model(X, W, b)
	print(A)
	return A >= 0.5
```

Ensuite, il suffit de fournir un nouveau couple $(x_1, x_2)$  à la fonction `predict`  avec notre modèle.

On peut même tracer la frontière de décision à partir de son équation (ensemble des points $(x_1, x_2)$ tels que $w_1 x_1 + w_2 x_2 = 0$)
$\Rightarrow$ $x_2 = \frac{-w_1 x_1 - b}{w_2}$ 

![[frontiere_decision.png]]

Le code développé peut bien sûr être utilisé pour des datasets avec autant de variables que l'on souhaite.

##### Problèmes

En travaillant sur un autre exemple (classification des photos de chiens et chats), on se rend compte qu'un certain nombre d'erreurs apparaît. 

Tout d'abord, les photos étant définies sur deux dimensions, on dispose à présent de données d'entraînement et de test sur 3 dimensions $\Rightarrow$ un `reshape` est nécessaire pour repasser à des tableaux à deux dimensions.

Ensuite, on s'aperçoit que la fonction exponentielle dans la sigmoïde peut rencontrer des problèmes d'overflow (la valeur renvoyée est trop grande pour être représentée en mémoire) $\Rightarrow$ dans ce cas, lorsque la sigmoïde renvoie 0 on rencontre des erreurs dans les logarithmes du Log Loss.
Pour éviter ça, on ajoute un petit $\epsilon = 1\text{e}^{-15}$ dans les $\log$.

D'autre part, il faut normaliser les données car cela évite l'overflow dans l'exponentielle et cela permet une bonne convergence de la descente de gradient (sinon compression de la fonction coût et rebondissements).

Une manière de normaliser est la normalisation MinMax :
$$X = \frac{X-X_{min}}{X_{max} - X_{min}}$$

Il faut également considérer la question des hyperparamètres. Il doivent en effet être ajustés (et lors des boucles où l'on calcule en vue de visualiser, ajouter une condition permettant de ne pas trop ralentir le code - toutes les 10 itérations par exemple et aussi mettre une barre de progression).

Élément à surveiller : l'over-fitting.
En effet, si le modèle devient trop performant sur les données d'entraînement, il peut finir par ne plus être capable de généraliser. Pour détecter ça, on trace les courbes sur le set d'entraînement mais aussi sur celui de test.

Comment remédier à l'over-fitting :
- fournir plus de données
- éviter un gros décalage entre le nombre de données et le nombre de variables (cf. le fléau de la dimension)
- utiliser une technique de régularisation

Mais plus généralement, un seul neurone n'est pas suffisant pour traiter de vrais problèmes : c'est un modèle linéaire.

---
### Réseau de neurones (2 couches)

Les régressions logitisques (neurones artificiels) sont parfaits pour séparer deux classes de points linéairement séparables. Si ce n'est pas le cas, ce n'est plus suffisant car le modèle est biaisé.

#### Remédier aux limitations

On peut passer au carré les variables existantes et les ajouter comme nouvelles variables. On obtient un modèle polynomial (feature engineering).
Une autre manière de faire (Deep Learning) est d'ajouter des neurones : laisser la machine apprendre à faire son feature engineering.

Ici c'est la seconde méthode qu'on va explorer en mettant en place un réseau de neurones à deux couches.

#### Couche 1

Dans cette première couche, on ajoute un second neurone tel qu'il ne partage pas de connexions avec le premier. Ainsi, chacun aura des paramètres $w_1$, $w_2$ et $b$ différents.
An niveau des notations, on a à présent : $w_{ij}$ paramètre associé au neurone $i$ et provenant de l'entrée $j$ et $b_i$ biais associé au neurone $i$.
Ainsi, pour un neurone $i$, on a une expression de la forme : 
$$z_i = w_{i1}x_1 + w_{i2}x_2 + b_i$$
On peut de cette manière avoir autant de neurones indépendants dans une couche. Plus on en aura, plus ce sera puissant mais lent.

#### Couche 2

On peut ensuite envoyer les résultats vers une seconde couche. Comme on a plusieurs couches, on met à jour les notations en faisait apparaître le numéro de la couche $k$ au niveau des variables :
$$z_i^{[k]} = w_{i1}^{[k]}x_1 + w_{i2}^{[k]}x_2 + b_i^{[k]}$$
Les entrées de la couche 2 sont les activations de la couche 1.

![[2_couches.png]]

On peut encore une fois rajouter autant de neurones qu'on veut dans cette deuxième couche (qui seront indépendants entre eux) mais aussi autant de couches qu'on veut.

Plus le réseau est profond (nombre de couches), plus il sera puissant mais lent. Il faut trouver un juste équilibre.

#### Vectorisation

Avec tous ces neurones, on ne va pas écrire les équations pour chaque neurone : c'est fastidieux et parfois impossible. On utilise la vectorisation à la place : chaque couche est représentée par une matrice.

Dans une couche, on réunit les résultats de tous les neurones dans une seule matrice $Z$ en modifiant les matrices $W$ et $b$ de manière à ce qu'elles aient autant de colonnes qu'il y a de neurones. $Z$ a donc autant de colonnes qu'il y'a de neurones dans la couche. 
Une autre manière de faire exister : avoir autant de lignes que de neurones. Ça impose d'ajuster au niveau des dimensions et l'expression devient :
$$Z^{[k]} = W^{[k]} \cdot X + b^{[k]}$$
La matrice $Z$ est juste transposée et cela permet d'aligner les entrées avec leurs neurones respectifs sur un schéma.
On fait ensuite passer $Z$ par la fonction d'activation et c'est le résultat $A$ qui sera en entrée de la prochaine couche.

Ce calcul de la première vers la dernière couche est la **Forward Propagation**.

#### Back-Propagation

Pour entraîner le réseau de neurones, c'est analogue à l'entraînement d'un seul neurone.

1. Définir une Fonction Coût
2. Calculer les dérivées partielles (par rapport aux paramètres de chaque couche)
3. Mettre à jour les paramètres avec la Descente de Gradient

Le calcul des dérivées partielles se fait avec la technique de Back-Propagation :
on retrace comment évolue la fonction coût de la dernière couche jusqu'à la première.

On calcule donc d'abord les gradients de la dernière couche puis pour calculer les gradients de la couche juste avant on commence par les dérivées de la dernière couche et on fait le lien avec le facteur $\frac{\partial Z^[i+1]}{\partial A[i]}$.
On retrace le calcul de la dernière équation jusqu'à la première.
On introduit des notations sous la forme $dZi$ afin de simplifier l'écriture.

![[Gradient’s.jpg]]

Finalement, ça donne :

$$\begin{matrix} dZ2 = (A^{[2]} - y) \\ \frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{1}{m} dZ2 \cdot (A^{[1]})^{T} \\ \frac{\partial \mathcal{L}}{\partial b^{[2]}} = \frac{1}{m} \sum_{axe 1}dZ2 \\ dZ1 = (W^{[2]})^T \cdot dZ2 \times A^{[1]}(1 - A^{[1]}) \\ \frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{1}{m} dZ1 \cdot X^{T}  \\ \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \frac{1}{m} \sum_{axe 1}dZ1 \end{matrix}$$

---
### Réseau de neurones profond

À présent, l'idée est de généraliser l'implémentation à $n$ couches. Pour cela, on retrace le passage d'une couche à deux.

#### Initialisation des paramètres

On remarque que pour la $c^{ième}$ couche, $$\boxed{\begin{matrix} W^{[c]}  \in \mathbb{R}^{n^{[c]} \times n^{[c-1]}} \\ b^{[c]} \in \mathbb{R}^{n^{[c]} \times 1} \end{matrix}}$$
On peut donc initialiser autant de paramètres voulus à l'aide d'une boucle `for`.

#### Forward Propagation

Pour une couche, on doit utiliser les activations de la couche précédente. Ainsi, on a pour la couche $c$ :
$$\boxed{\begin{matrix} Z^{[c]}  = W^{[c]} \cdot A^{[c-1]} + b^{[c]} \\ A^{[c]} = \frac{1}{1 + e^{-Z^{[c]}}} \end{matrix}}$$

Pour la première couche c'est particulier (l'entrée est $X$). On considère $X = A^{[0]}$.
On utilise à nouveau une boucle `for` pour remplir un dictionnaire d'activation.

#### Back Propagation

Ici, pour la couche $c$ :
$$\boxed{\begin{matrix} dZ^{[C_f]}  = A^{[C_f]} - y \\ dW^{[c]} = \frac{1}{m} \times dZ^{[c]} \cdot A^{[c-1]^T} \\ db^{[c]} = \frac{1}{m} \sum_{axe1} dZ^{[c]} \\ dZ^{[c-1]} = W^{[c]^T} \cdot dZ^{[c]} \times A^{[c-1]}(1 - A^{[c-1]}) \end{matrix}}$$

Ici, on effectue une boucle `for` allant de la dernière couche à la première en calculant tous ces éléments successivement. 

#### Descente de Gradient

$$\begin{matrix} W^{[c]} = W^{[c]} - \alpha \times dW^{[c]} \\ b^{[c]} = b^{[c]} - \alpha \times db^{[c]} \end{matrix}$$

#### Discussion

Plus on a de couches et de neurones par couche, plus on est capable de réaliser des classifications complexes.
Cependant, il y'a aussi des risques : plus un réseau de neurones est profond, plus il a des chances de se perdre dans son apprentissage (Vanishing Gradients).