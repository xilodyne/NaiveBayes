package xilodyne.machinelearning.classifier.examples;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import xilodyne.machinelearning.classifier.NaiveBayesClassifier;
import xilodyne.util.G;
import xilodyne.util.Logger;


/**
 * Tests Naive Bayes using Gender (text) and Names (text).
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 * 
 */
public class NB_NameGender {

	private static Logger log = new Logger();

	public static void main(String[] args) {
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		G.setLoggerLevel(G.LOG_DEBUG);
		log.logln_withClassName(G.lF, "");

		List<String> labelList = new ArrayList<String>(Arrays.asList("Name", ">170cm", "Eye", "Hair"));
		List<String> classification = new ArrayList<String>(Arrays.asList("Male", "Female"));
		NaiveBayesClassifier nb = new NaiveBayesClassifier(classification, labelList);

		nb.fit(new ArrayList<String>(Arrays.asList("Drew", "No", "Blue", "Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Claudia", "Yes", "Brown", "Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew", "No", "Blue", "Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew", "No", "Blue", "Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Alberto", "Yes", "Brown", "Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Karin", "No", "Blue", "Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Nina", "Yes", "Brown", "Short")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Sergio", "Yes", "Blue", "Long")), "Male");

		System.out.println();
		nb.printFeaturesAndClasses();
		nb.predictUsingFeatureName(labelList.indexOf("Name"), "Drew");

		nb.determineProbabilities();
		nb.predictUsingFeatureName(labelList.indexOf("Hair"), "Long");

		nb.predictUsingFeatureNames(new ArrayList<String>(Arrays.asList("Drew", "Yes", "Blue", "Long")));
		System.out
				.println("Result: " + nb.predict(new ArrayList<String>(Arrays.asList("Drew", "Yes", "Blue", "Long"))));
	}

}