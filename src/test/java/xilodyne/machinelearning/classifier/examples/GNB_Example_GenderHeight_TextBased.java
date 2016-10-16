package xilodyne.machinelearning.classifier.examples;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.machinelearning.classifier.GaussianNB;

/**
 * Gaussian NB using gender (text) and float attributes
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 */
public class GNB_Example_GenderHeight_TextBased {

	private static Logger log = new Logger();

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
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		G.setLoggerLevel(G.LOG_DEBUG);
		log.logln_withClassName(G.lF,"");

		List<String> labelList = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)", "Ft(in)"));
		List<String> classification = new ArrayList<String>(Arrays.asList("Male", "Female"));

		GaussianNB gnb = new GaussianNB(GaussianNB.EMPTY_SAMPLES_ALLOW, classification, labelList);

		gnb.setClassListDisplayName("Gender");

		gnb.fit(new ArrayList<Float>(Arrays.asList(6f, 180f, 12f)), "Male");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f, 190f, 11f)), "Male");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.58f, 170f, 12f)), "Male");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f, 165f, 10f)), "Male");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5f, 100f, 6f)), "Female");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.5f, 150f, 8f)), "Female");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.42f, 130f, 7f)), "Female");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.75f, 150f, 9f)), "Female");

		gnb.printAttributeValuesAndClasses();
		gnb.printMeanVar();

		gnb.predictSingleLabelSingleClass(classification.indexOf("Male"), 0, 6.0f);
		String predictedResult = gnb.predict(new ArrayList<Float>(Arrays.asList(6f, 130f, 8f)));
		System.out.println("CLASS Prediction: " + predictedResult);
	}
}
