 # Naive Bayes / Gaussian Naive Bayes

Java implementation of Naive Bayes ([as described by Prof Eamonn Keogh, UCR](http://www.cs.ucr.edu/~eamonn/CE/Bayesian%20Classification%20withInsect_examples.pdf)) and Gaussian Naive Bayes ([as described by wikipedia](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes) ).


## NDArray

These routines make use of the NDArray from Mike Anderson ([GitHub](https://github.com/mikera/vectorz)).


## Required libraries

The xilodyne util libraries are required ([GitHub](https://github.com/xilodyne/xilodyne.util)).

```
xilodyne-util-array-bin.jar
xilodyne-util-data-bin.jar
xilodyne-util-fileio-bin.jar
xilodyne-util-logger-bin.jar
xilodyne-util-metrics-bin.jar
xilodyne-util-weka_helper-bin.jar
```

Maven requirements (add to your pom.xml or download manually)

```
<dependency>
    <groupId>net.mikera</groupId>
    <artifactId>vectorz</artifactId>
    <version>0.62.0</version>
</dependency>
<dependency>
    <groupId>de.siegmar</groupId>
    <artifactId>fastcsv</artifactId>
    <version>1.0.1</version>
</dependency
<dependency>
    <groupId>junit</groupId>
    <artifactId>junit</artifactId>
    <version>4.12</version>
    <scope>test</scope>
</dependency>
```

If the example files are to be run, then also needed is:

```
<dependency>
    <groupId>nz.ac.waikato.cms.weka</groupId>
    <artifactId>weka-stable</artifactId>
    <version>3.8.0</version>
</dependency>
```


# License

NB & GNB are licensed under the MIT License ([link](https://opensource.org/licenses/MIT)).  Other component and libraries licenses are found in the doc directory.


# Authors

**Austin Davis Holiday** - *Initial work* 

I can be reached at [aholiday@xilodyne.com](mailto:aholiday@xilodyne.com)
