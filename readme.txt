BEGONIN Florian 11400915
GRANDJEAN Valentin 11402835

AM3 - Deep Learning

Objectifs :

L'execution du fichier mnist.py permet d'entrainer un réseau neuronal convolutif sur la base de données MNIST.
La base de données sera automatiquement téléchargée lors de l'execution du programme si celle ci est manquante.

L'execution du fichier classificationCifar10.py permet d'entrainer un réseau neuronal convolutif sur la base de
données CIFAR 10, celle ci doit être présente dans le dossier et ne sera pas téléchargée automatiquement si elle
est manquante. Cependant ce programme est une base devant être améliorée pour atteindre un résultat convenable.

Mnist.py :

Après la phase d'entrainement des réseaux neuronaux il est possible de vérifier les résultats obtenus.
Taper 1 dans la console testera le classifier en lui transmettant une image. Le résultat affiché
correspond à l'image ayant été choisie dans la base de donnée, son label ainsi que les prédictions du
réseau.
Taper 2 dans la console testera le réseau inverse en lui demandant de créer une représentation de chaque
chiffre puis passera cette image dans le classifier pour valider la reconaissance de chaque image.
Taper 3 mettra fin à l'execution du programme.

Installation :

Nécessite l'installation de tensorflow, numpy et matplotlib.

