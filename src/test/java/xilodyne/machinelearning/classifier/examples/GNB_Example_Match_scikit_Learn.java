package xilodyne.machinelearning.classifier.examples;

import mikera.arrayz.NDArray;


/**
 * Verify Gaussian Naive Bayes produces same results as 
 * scikit-learn GNB.
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.2 -- Changes to reflect v.02 GNB update
 * 
 */

import xilodyne.machinelearning.classifier.bayes.GaussianNaiveBayesClassifier;
import xilodyne.util.ArrayUtils;
import xilodyne.util.logger.Logger;

/**
 * @author Austin Davis Holiday (aholiday@xilodyne.com)
 * @version 0.4 - 1/30/2018 - reflect xilodyne util changes
 *
 */
public class GNB_Example_Match_scikit_Learn {

	private static Logger log = new Logger("egnb");

	public static void main(String[] args) {
		// Logger.setLoggerLevel(Logger.LOG_OFF);
		Logger.setLoggerLevel(Logger.LOG_FINE);
		// Logger.setLoggerLevel(Logger.LOG_INFO);
		//Logger.setLoggerLevel(Logger.LOG_DEBUG);
		log.logln_withClassName(Logger.lF, "");

		log.logln_withClassName(Logger.lD, "");

		NDArray trainingData = NDArray.newArray(6, 2);
		NDArray testingData1 = NDArray.newArray(2, 2);
		NDArray testingData2 = NDArray.newArray(1, 2);
		double[] trainingLabels;

		System.out.println();
		System.out.println();
		System.out.println("*** TEST *** Check SKLearn values");

		/*
		 * >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3,
		 * 2]]) >>> Y = np.array([1, 1, 1, 2, 2, 2]) >>> from
		 * sklearn.naive_bayes import GaussianNB >>> clf = GaussianNB() >>>
		 * clf.fit(X, Y) GaussianNB() >>> print(clf.predict([[-0.8, -1]])) [1]
		 */
		trainingData.set(0, 0, -1);
		trainingData.set(0, 1, -1);

		trainingData.set(1, 0, -2);
		trainingData.set(1, 1, -1);

		trainingData.set(2, 0, -3);
		trainingData.set(2, 1, -2);

		trainingData.set(3, 0, 1);
		trainingData.set(3, 1, 1);

		trainingData.set(4, 0, 2);
		trainingData.set(4, 1, 1);

		trainingData.set(5, 0, 3);
		trainingData.set(5, 1, 2);

		trainingLabels = new double[] { 1, 1, 1, 2, 2, 2 };

		GaussianNaiveBayesClassifier gnb = new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW);
		try {
			gnb.fit(trainingData, trainingLabels);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();
		testingData1.set(0, 0, -0.8);
		testingData1.set(0, 1, -1);
		testingData1.set(1, 0, 1);
		testingData1.set(1, 1, 2);

		double[] result = gnb.predict(testingData1);


		testingData2.set(0, 0, -0.8);
		testingData2.set(0, 1, -1);
		System.out.println("Testing Data: " + testingData2);
		double[] scores = gnb.getProbabilityScores_TestingSet(testingData2);
		System.out.println("Label Scores: " + ArrayUtils.printArrayShowFull(scores));

		testingData2.set(0, 0, 1);
		testingData2.set(0, 1, 2);
		System.out.println("Testing Data: " + testingData2);
		scores = gnb.getProbabilityScores_TestingSet(testingData2);
		System.out.println("Label Scores: " + ArrayUtils.printArrayShowFull(scores));
		
		double accuracy = gnb.getAccuracyOfPredictedResults(new double[] { 1.0 }, result);
		System.out.println("\nTesting Data: " + testingData1);
		System.out.println("Results: " + ArrayUtils.printArray(result));


		System.out.println("Accuracy: " + accuracy);
		System.out.println();

	}

}
