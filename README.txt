USAGE:
python main.py type max_pass
where type (required) specifies which kind of perceptron to run, and max_pass is optional (default is 500)

Type options:
1 - Binary perceptron on spambase dataset
2 - Multiclass perceptron using one-vs-rest reduction, on activity dataset
3 - Multiclass perceptron using one-vs-one reduction, on activity dataset
4 - Direct multiclass perceptron on activity dataset

* Binary perceptron uses the spambase dateset: https://archive.ics.uci.edu/ml/datasets/spambase
* Multiclass uses the activity dataset: https://archive.ics.uci.edu/ml/datasets/Activity+recognition+using+wearable+physiological+measurements
* Uses pandas, numpy and matplotlib libraries.
* Implemented and tested using python 3.10
* all exercises should finish in under 2 minutes for max_pass=500
