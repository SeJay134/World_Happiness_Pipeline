python version 3.13.13

Project: World Happiness Pipeline
The prefect pipeline that performs an end-to-end analysis of the World Happiness dataset

Sergei Patrushev
--------------------------------------------------------------------------------------
Pearson correlation coefficient is a statistical measure that quantifies the strength and direction of 
the linear relationship between two variables. Its values range from −1 to +1, where:

+1 indicates a perfect positive linear relationship,
−1 indicates a perfect negative linear relationship,
0 indicates no linear relationship.

In simple terms, it measures how closely the points in a scatter plot follow a straight line.

import numpy as np
r = np.corrcoef(x, y)[0, 1]
---------------------------------------------------------------------------------------