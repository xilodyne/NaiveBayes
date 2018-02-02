package xilodyne.machinelearning.classifier.examples;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import xilodyne.util.logger.Logger;
import xilodyne.machinelearning.classifier.bayes.GaussianNaiveBayesClassifier;

/**
 * Gaussian NB using gender features
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.4 - 1/30/2018 - reflect xilodyne util changes
 * @version 0.2  
 */
public class GNB_Example_GenderHeight_TextBased {

	private static Logger log = new Logger("egnb");

	/*
	 * Gender height (feet) weight (lbs) foot size(inches) 
	 * male 6 180 12 
	 * male 5.92 (5'11") 190 11 
	 * male 5.58 (5'7") 170 12 
	 * male 5.92 (5'11") 165 10
	 * female 5 100 6 
	 * female 5.5 (5'6") 150 8 
	 * female 5.42 (5'5") 130 7 
	 * female 5.75 (5'9") 150 9
	 */

	public static void main(String[] args) {
		// Logger.setLoggerLevel(Logger.LOG_OFF);
		// Logger.setLoggerLevel(Logger.LOG_FINE);
		// Logger.setLoggerLevel(Logger.LOG_INFO);
		Logger.setLoggerLevel(Logger.LOG_DEBUG);
		log.logln_withClassName(Logger.lF,"");

		List<String> featureNames = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)", "Ft(in)"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male", "Female"));

		int indexMale = featureNames.indexOf("Male");
		int indexFemale = labelNames.indexOf("Female");

		GaussianNaiveBayesClassifier gnb = new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);

		gnb.setLabelClassCategory("Gender");

		gnb.fit(new ArrayList<Double>(Arrays.asList(6.0, 180.0, 12.0)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f, 190f, 11f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.58f, 170f, 12f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f, 165f, 10f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5f, 100f, 6f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.5f, 150f, 8f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.42f, 130f, 7f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.75f, 150f, 9f)), indexFemale);

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();

		gnb.getProbabilty_OneFeature(0, labelNames.indexOf("Male"),  6.0f);
		double predictedResult = gnb.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f, 130f, 8f)));
		System.out.println("CLASS Prediction: " + labelNames.get((int)predictedResult));
	}
}
