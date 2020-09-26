Keywords: Python, Clustring, K-means, K-median, Optimization, Pulp, Cplex, Animated Plot, Mixed Integer Linear Programming
## Find similar wines!
The purpose of this task is to cluster the wines based of 13 following features using kmeans algorithm. The data provided with labels (Type column), so we keep in mind to avoid 'Type' contributing in clustring since we are dealing with unsupervised learning.

0. Type
1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline

### What might pique your interests
However the purpose of this project is clustring, you will find the k-median optimization problem modeled as Mixed Integer Linear Programming amazing. The python package <a href='https://pypi.org/project/PuLP/'> pulp </a> with Cplex core engine is an open source available package for python.
Also, have you ever searched for animated plot in jupyter notebook? Download <a href='https://github.com/MosesDastmard/wine_clustring/blob/master/clustring.ipynb'>clustring.ipynb</a>. You will see how dynamically the clusters plot updated during k-means optimization.

### Scaling
As far as we concern in clustring based on the distance of the points, the distance is sensitive to the unit of measurment. So for those features with higher unit of measurment contribute the most bias the distance to higher values and dominate those with less unit of measurment. So first step that must be taken is scaling the features to have the same unit of measurment.

### visualizing
Since we are dealing with 13 features that is not possible to be plotted on 2D screen. we need to reduce the dimentionallity. Principle Component Anaysis (PCA) is one the most common-used dimension reduction precedure that turn the points in such way that the first components explain the most possible variance of the points. So taking the first two components makes sense to visualize the point.


```python
import clustering_lib
import pandas as pd
```


```python
label_name = ['Type']
feature_names = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
columns_name = label_name + feature_names
wine_df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names = columns_name)
```


```python
wine_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Alcohol</th>
      <th>Malic acid</th>
      <th>Ash</th>
      <th>Alcalinity of ash</th>
      <th>Magnesium</th>
      <th>Total phenols</th>
      <th>Flavanoids</th>
      <th>Nonflavanoid phenols</th>
      <th>Proanthocyanins</th>
      <th>Color intensity</th>
      <th>Hue</th>
      <th>OD280/OD315 of diluted wines</th>
      <th>Proline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>14.23</td>
      <td>1.71</td>
      <td>2.43</td>
      <td>15.6</td>
      <td>127</td>
      <td>2.80</td>
      <td>3.06</td>
      <td>0.28</td>
      <td>2.29</td>
      <td>5.64</td>
      <td>1.04</td>
      <td>3.92</td>
      <td>1065</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>13.20</td>
      <td>1.78</td>
      <td>2.14</td>
      <td>11.2</td>
      <td>100</td>
      <td>2.65</td>
      <td>2.76</td>
      <td>0.26</td>
      <td>1.28</td>
      <td>4.38</td>
      <td>1.05</td>
      <td>3.40</td>
      <td>1050</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>13.16</td>
      <td>2.36</td>
      <td>2.67</td>
      <td>18.6</td>
      <td>101</td>
      <td>2.80</td>
      <td>3.24</td>
      <td>0.30</td>
      <td>2.81</td>
      <td>5.68</td>
      <td>1.03</td>
      <td>3.17</td>
      <td>1185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>14.37</td>
      <td>1.95</td>
      <td>2.50</td>
      <td>16.8</td>
      <td>113</td>
      <td>3.85</td>
      <td>3.49</td>
      <td>0.24</td>
      <td>2.18</td>
      <td>7.80</td>
      <td>0.86</td>
      <td>3.45</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>13.24</td>
      <td>2.59</td>
      <td>2.87</td>
      <td>21.0</td>
      <td>118</td>
      <td>2.80</td>
      <td>2.69</td>
      <td>0.39</td>
      <td>1.82</td>
      <td>4.32</td>
      <td>1.04</td>
      <td>2.93</td>
      <td>735</td>
    </tr>
  </tbody>
</table>
</div>



### How many clusters
Finding the right number of clusters is not a easy, but the elbow method is a way to guess the number of clusters based on elbow point that is shown as red dot (3 clusters) in the following plot. 


```python
SSD = list()
for k in range(1,10):
    k_mean_wine = clustering_lib.k_mean(data = wine_df, K = k, features = feature_names)
    results = k_mean_wine.apply()
    SSD.append(results['SSD'][-1])
```


```python
import matplotlib.pyplot as plt
plt.plot(range(1,10),SSD)
plt.xlabel('Numeber of clusters')
plt.ylabel('Cost')
plt.scatter(3,SSD[2], color = 'r')
```




    <matplotlib.collections.PathCollection at 0x17924157198>




![png](output_6_1.png)



```python
k_mean_wine = clustering_lib.k_mean(data = wine_df, K = 3, features = feature_names)
```


```python
k_mean_wine.data_df.columns
k_mean_wine.scaled_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Alcohol_scaled</th>
      <th>Malic acid_scaled</th>
      <th>Ash_scaled</th>
      <th>Alcalinity of ash_scaled</th>
      <th>Magnesium_scaled</th>
      <th>Total phenols_scaled</th>
      <th>Flavanoids_scaled</th>
      <th>Nonflavanoid phenols_scaled</th>
      <th>Proanthocyanins_scaled</th>
      <th>Color intensity_scaled</th>
      <th>Hue_scaled</th>
      <th>OD280/OD315 of diluted wines_scaled</th>
      <th>Proline_scaled</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.518613</td>
      <td>-0.562250</td>
      <td>0.232053</td>
      <td>-1.169593</td>
      <td>1.913905</td>
      <td>0.808997</td>
      <td>1.034819</td>
      <td>-0.659563</td>
      <td>1.224884</td>
      <td>0.251717</td>
      <td>0.362177</td>
      <td>1.847920</td>
      <td>1.013009</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.246290</td>
      <td>-0.499413</td>
      <td>-0.827996</td>
      <td>-2.490847</td>
      <td>0.018145</td>
      <td>0.568648</td>
      <td>0.733629</td>
      <td>-0.820719</td>
      <td>-0.544721</td>
      <td>-0.293321</td>
      <td>0.406051</td>
      <td>1.113449</td>
      <td>0.965242</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.196879</td>
      <td>0.021231</td>
      <td>1.109334</td>
      <td>-0.268738</td>
      <td>0.088358</td>
      <td>0.808997</td>
      <td>1.215533</td>
      <td>-0.498407</td>
      <td>2.135968</td>
      <td>0.269020</td>
      <td>0.318304</td>
      <td>0.788587</td>
      <td>1.395148</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.691550</td>
      <td>-0.346811</td>
      <td>0.487926</td>
      <td>-0.809251</td>
      <td>0.930918</td>
      <td>2.491446</td>
      <td>1.466525</td>
      <td>-0.981875</td>
      <td>1.032155</td>
      <td>1.186068</td>
      <td>-0.427544</td>
      <td>1.184071</td>
      <td>2.334574</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.295700</td>
      <td>0.227694</td>
      <td>1.840403</td>
      <td>0.451946</td>
      <td>1.281985</td>
      <td>0.808997</td>
      <td>0.663351</td>
      <td>0.226796</td>
      <td>0.401404</td>
      <td>-0.319276</td>
      <td>0.362177</td>
      <td>0.449601</td>
      <td>-0.037874</td>
    </tr>
  </tbody>
</table>
</div>




```python
results = k_mean_wine.apply()
```


```python
import matplotlib.pyplot as plt
ani = clustering_lib.SubplotAnimation(results)
```

<img src="Animated.png">

The first plot, made using the first two components of PCA, show us that the points are devided in a good way in k different clusters. There are not overlapping between different clusters.
The second plot show that after 5 iteration the SSD converge, this means that the distance between the centroids at the step 4 and the centroinds at the step 5 is less than a prefixed treshold.

###  Features contribute the most


```python
import seaborn as sns
import pandas as pd
sns.pairplot(results['data'][feature_names+["argmin"]], hue="argmin", height=2, vars = feature_names, palette = {0.0:'black',1.0:'red',2.0:'blue'})#,3.0:'green'})
plt.show() 
```

<img src="corr_plot.png">

The previous plots are really usefull to see which features  contribute more to  find the clusters.
If we look at the histograms on the diagonal we can understand if the clusters are able to distinguish the clusters using a given feature.

As we can see in the following plot, if the histogram along a direction is unimodal this means that there is no way with k-means to distinguish the different points using that particular characteristic. only if the distribution of the points is multimodal along one direction, then those features will contribute to the division into clusters.

<img src="clust.png">

Looking at the distributions of 'Magnesium' or 'Ash', we can see how the three different histograms are one above the other, this means that these characteristics are not significant for  distinguishing the clusters. 

On the contrary some features like 'Alcohol' or 'Flavanoids' have very distinct histograms, this means that these features contribute a lot to the formation of clusters.

From the file wine.names we can read that the wines all come from the same region in Italy but there are three different cultivars.
The quantity of flavonoids present in the wine, as can be read in this [link](http://lem.ch.unito.it/didattica/infochimica/2007_Polifenoli_Vino/flavo.html), are greater in grapes ripened in soils with a high exposure to the sun and in the respective wines.
So all the more so these Flavonoids are important in the division into clusters.

## K-means can go wrong!
K-means picks initial centroid randomly from data points that make it likely to get trapped in a local minimum. For example for $K= 3$ if in the initialization step, K-means picks the centroids that are close to each other, it is more likely to reach a local minimum rather than global. The probability of finding a local minimum decreases using an algorithm like K-means++, however, it works better than K-means but it can not guarantee global minimum.

### K-median
K-median is a problem in operation research science that is about allocating the points to the nearest median. K-median is similar to clustering problem with cost function as a summation of squared distance. K-median in terms of a mathematical formulation is of the mixed-integer linear programming (MILP) if the solution space narrowed down to the points space.

There exists an algorithm named branch & bound (B&B) that guarantees reaching the global minimum for MILP. This algorithm uses simplex to get the initial solution by relaxing the constraint of integer variables, then in each brach one variable set to closest integer till all variable set to an integer value and it drops the subproblems by bounding that can not improve its current solution. (more [details](https://en.wikipedia.org/wiki/Branch_and_bound))

### Mathematical formulation
define:

<img src="mathematical_formulation.png">


```python
import theoretical_lib
Global_minimum = theoretical_lib.k_median(k_mean_wine.scaled_df, K = 3)
```


```python
print("Global minimum is", Global_minimum, "for", 3, "clusters")
```

    Global minimum is 1564.606349269607 for 3 clusters
    

### Conclusion
The designed K-means algorithm picks the initial centroids arbitrarily for the wine dataset, and it returns centroids with the cost of around 2370, while the equivalent K-median problem returns cost of 1564 on the solution space. It proves that K-means failed to reach the global minimum, however finding the solution and doesn't guarantee the global optimum. The main reason makes K-mean popular is its speed. It should be mentioned that, however, K-median MILP can lead to better solution but it is still too slow.   


```python
theoretical_lib.plot_res()
```


![png](output_20_0.png)

