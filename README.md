##Electrical Engineering 300 Project

This project uses Bayesian Multivariate Linear Regression to predict miles per gallon from a data set of 392 cars.
It uses a weight to adjust the coefficients on which the predictions are made. Those are input as arguments when running the predictMPG.py file and printEquation.py. So, if you wanted to run those files with an optimal weight of 41 it would look like...

    >python predictMPG.py 41
    >python printEquation.py 41

If you leave these blank it  will result to 1. As for the other files, they are run normally so in order to set those it is required to have Python 3.4+, numpy and matplotlib. If you have those then run the following commands.

    >git clone
    >cd MPG-Prediction-Model

That's it! If you want to see the report send me an email, although as of 10/26/2016, it is a work in progress.

Data Set:
Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
