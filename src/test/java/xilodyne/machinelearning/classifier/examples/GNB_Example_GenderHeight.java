package xilodyne.machinelearning.classifier.examples;

import java.util.ArrayList;
import java.util.Arrays;

import mikera.arrayz.NDArray;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.machinelearning.classifier.bayes.GaussianNaiveBayesClassifier;


/**
 * Gaussian NB using gender (float) and float attributes
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 */
public class GNB_Example_GenderHeight {

	private static Logger log = new Logger();

	public static void main(String[] args) {
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		G.setLoggerLevel(G.LOG_DEBUG);
		log.logln_withClassName(G.lF,"");

		NDArray trainingData = NDArray.newArray(8, 2);
		NDArray testingData = NDArray.newArray(1,2);

		double[] trainingLabels;
		GaussianNaiveBayesClassifier gnb = new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW);

		/**
		 * CLASS
		 * male = 0.0 
		 * female = 1.0 
		 * 
		 * ATTRIBUTES
		 * Gender, height (feet), weight (lbs) 
		 * male 6 180
		 * male 5.92 (5'11") 190 
		 * male 5.58 (5'7") 170 
		 * male 5.92 (5'11") 165
		 * female 5 100 
		 * female 5.5 (5'6") 150 
		 * female 5.42 (5'5") 130 
		 * female 5.75 (5'9") 150
		 */

		/** Create NDArray of [height, weight] */
		
		trainingData.set(0, 0, 6.0);
		trainingData.set(0, 1, 180.0);

		trainingData.set(1, 0, 5.92);
		trainingData.set(1, 1, 190.0);

		trainingData.set(2, 0, 5.58);
		trainingData.set(2, 1, 170.0);

		trainingData.set(3, 0, 5.92);
		trainingData.set(3, 1, 165.0);

		trainingData.set(4, 0, 5.5);
		trainingData.set(4, 1, 100.0);

		trainingData.set(5, 0, 5.5);
		trainingData.set(5, 1, 150.0);

		trainingData.set(6, 0, 5.42);
		trainingData.set(6, 1, 130.0);

		trainingData.set(7, 0, 5.75);
		trainingData.set(7, 1, 150.0);

		/** label each corresponding entry with male or female */
		trainingLabels = new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 };


		try {
			gnb.fit(trainingData, trainingLabels);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();
		
		double results = gnb.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f, 130f)));
		System.out.println("\n\n **** Using Float List.");
		System.out.println("Label Predicted: " + results);
		
		double[] scores = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		System.out.println("\n\nScore label 0: " + scores[0]);
		System.out.println("Score label 1: " + scores[1]);
		
		
		//use ndarray with one entry
		results = -1;
		scores = null;
		
		testingData.set(0,0,6.0);
		testingData.set(0,1,130.0);
		
		results = gnb.predict_TestingSet(testingData);
		System.out.println("\n\n **** Using NDArray, single entry.");
		System.out.println("Label Predicted: " + results);
		
		scores = gnb.getProbabilityScores_TestingSet(testingData);
		System.out.println("\n\nScore label 0: " + scores[0]);
		System.out.println("Score label 1: " + scores[1]);

		
		

	}

}
