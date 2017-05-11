package xilodyne.machinelearning.classifier.examples;


import java.util.Hashtable;

import xilodyne.machinelearning.classifier.bayes.NaiveBayesClassifier_UsingTextValues;
import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;


/**
 * Tests Naive Bayes using Gender (text) and Names (text).
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.2
 * 
 */
public class NB_TextBased_NameGender {

	private static Logger log = new Logger();

	public static void main(String[] args) {
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		G.setLoggerLevel(G.LOG_DEBUG);
		log.logln_withClassName(G.lF, "");

		String[] featureCategories = new String[]{"Name",">170cm","Eye","Hair"};
		String[] labels = new String[]{"Male","Female"};

		NaiveBayesClassifier_UsingTextValues nb = new NaiveBayesClassifier_UsingTextValues(NaiveBayesClassifier_UsingTextValues.EMPTY_SAMPLES_IGNORE);

		nb.fit(featureCategories[0], "Drew", labels[0]);
		nb.fit(featureCategories[1], "No", labels[0]);
		nb.fit(featureCategories[2], "Blue", labels[0]);
		nb.fit(featureCategories[3], "Short", labels[0]);
		
		nb.fit(featureCategories[0], "Claudia", labels[1]);
		nb.fit(featureCategories[1], "Yes", labels[1]);
		nb.fit(featureCategories[2], "Brown", labels[1]);
		nb.fit(featureCategories[3], "Long", labels[1]);

		nb.fit(featureCategories[0], "Drew", labels[1]);
		nb.fit(featureCategories[1], "No", labels[1]);
		nb.fit(featureCategories[2], "Blue", labels[1]);
		nb.fit(featureCategories[3], "Long", labels[1]);
		
		nb.fit(featureCategories[0], "Drew", labels[1]);
		nb.fit(featureCategories[1], "No", labels[1]);
		nb.fit(featureCategories[2], "Blue", labels[1]);
		nb.fit(featureCategories[3], "Long", labels[1]);
		
		nb.fit(featureCategories[0], "Alberto", labels[0]);
		nb.fit(featureCategories[1], "Yes", labels[0]);
		nb.fit(featureCategories[2], "Brown", labels[0]);
		nb.fit(featureCategories[3], "Short", labels[0]);
		
		nb.fit(featureCategories[0], "Drew", labels[1]);
		nb.fit(featureCategories[1], "No", labels[1]);
		nb.fit(featureCategories[2], "Blue", labels[1]);
		nb.fit(featureCategories[3], "Long", labels[1]);
		
		nb.fit(featureCategories[0], "Karin", labels[1]);
		nb.fit(featureCategories[1], "Yes", labels[1]);
		nb.fit(featureCategories[2], "Brown", labels[1]);
		nb.fit(featureCategories[3], "Short", labels[1]);
		
		nb.fit(featureCategories[0], "Sergio", labels[0]);
		nb.fit(featureCategories[1], "Yes", labels[0]);
		nb.fit(featureCategories[2], "Blue", labels[0]);
		nb.fit(featureCategories[3], "Long", labels[0]);

		System.out.println();
		nb.printFeaturesAndLabels();
		
		String predictedLabel = null;
		Hashtable<String, String> testingData_OneSet = new Hashtable<String, String>();
		testingData_OneSet.put(featureCategories[0], "Drew");
		testingData_OneSet.put(featureCategories[1], "Yes");
		testingData_OneSet.put(featureCategories[2], "Blue");
		testingData_OneSet.put(featureCategories[3], "Long");
		predictedLabel = nb.predict_TestingSet(testingData_OneSet);
		System.out.println("Given: " + ArrayUtils.printArray(testingData_OneSet));
		System.out.println("Predicted: " + predictedLabel);

		double[] results = nb.getProbabilityScores_TestingSet(testingData_OneSet);
		System.out.println("Probabilty male: " + results[0]);
		System.out.println("Probabilty female: " + results[1]);

	}

}