package xilodyne.machinelearning.classifier.test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


import mikera.arrayz.NDArray;

import org.junit.Test;

import xilodyne.machinelearning.classifier.bayes.GaussianNaiveBayesClassifier;
import xilodyne.machinelearning.classifier.io.AccessSerializedObject;
import xilodyne.util.ArrayUtils;
import xilodyne.util.logger.Logger;

/**
 * JUnit tests for the Gaussian Naive Bayes.
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.4 - 1/30/2018 - reflect xilodyne util changes
 * @version 0.2 -- Changes to reflect v.02 GNB update
 * 
 */


public class Test_GaussianNB {
	private Logger log = new Logger("tGNB");
	
	public Test_GaussianNB() {
		// Logger.setLoggerLevel(Logger.LOG_OFF);
		Logger.setLoggerLevel(Logger.LOG_FINE);
		// Logger.setLoggerLevel(Logger.LOG_INFO);
		//Logger.setLoggerLevel(Logger.LOG_DEBUG);
	}
	
	@Test
	public void checkProbability_OneFeature() {
		log.logln_withClassName(Logger.lF,"");	
			
		//setup
		System.out.println();
		System.out.println("*** TEST *** Check Probability One Label");
		System.out.println("Testing:  P(Male|Height)");
		System.out.println();
		
		List<String> featureNames = new ArrayList<String>(Arrays.asList("Height"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male","Female"));
		// 0 = Male
		// 1 = Female
		
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);
		gnb.setLabelClassCategory("Gender");
		
		int labelIndex = featureNames.indexOf("Height");
		int maleIndex = labelNames.indexOf("Male");
		int indexFemale = labelNames.indexOf("Female");
		
		gnb.fit(labelIndex, 6f, maleIndex);
		gnb.fit(labelIndex, 5.92f, maleIndex);
		gnb.fit(labelIndex, 5.58f, maleIndex);
		gnb.fit(labelIndex, 5.92f, maleIndex);
		gnb.fit(labelIndex, 5f, indexFemale);
		gnb.fit(labelIndex, 5.5f, indexFemale);
		gnb.fit(labelIndex, 5.42f, indexFemale);
		gnb.fit(labelIndex, 5.75f, indexFemale);

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();
		
		float result = gnb.getProbabilty_OneFeature(featureNames.indexOf("Height"), labelNames.indexOf("Male"), 6.0f);
		assertEquals(0.7894416, result, 0.000001);
		
		result = gnb.getProbabilty_OneFeature(featureNames.indexOf("Height"), labelNames.indexOf("Female"),  6.0f);
		assertEquals(0.11172937, result, 0.000001);
		System.out.println("*** TEST COMPLETE ***");
	}

	@Test
	public void checkProbability_ThreeFeatures() {
		log.logln_withClassName(Logger.lF,"");

		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Three Samples");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs, 8\" shoe)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs, 8\" shoe)");

		List<String> featureNames = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)","Ft(in)"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male","Female"));
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);

		int indexMale = labelNames.indexOf("Male");
		int indexFemale = labelNames.indexOf("Female");
		
		gnb.setLabelClassCategory("Gender");

		/*
		 Gender height (feet) weight (lbs) foot size(inches) 
		 male 6 180 12 
		 male 5.92 (5'11") 190 11 
		 male 5.58 (5'7") 170 12 
		 male 5.92 (5'11") 165 10 
		 female 5 100 6 
		 female 5.5 (5'6") 150 8 
		 female 5.42 (5'5") 130 7 
		 female 5.75 (5'9") 150 9 
		 */
		gnb.fit(new ArrayList<Float>(Arrays.asList(6f,180f,12f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,190f,11f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.58f,170f,12f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,165f,10f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5f,100f,6f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.5f,150f,8f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.42f,130f,7f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.75f,150f,9f)), indexFemale);

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();

		double[] results = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f,8f)));
		
		assertEquals(6.1970717E-9, results[0], 0.000001);
		assertEquals(5.37791E-4, results[1], 0.00001);

		System.out.println("*** TEST COMPLETE ***");
	}
	@Test
	public void checkProbability_ThreeFeatures_FloatArray() {
		log.logln_withClassName(Logger.lF,"");

		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Three Samples");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs, 8\" shoe)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs, 8\" shoe)");

		List<String> featureNames = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)","Ft(in)"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male","Female"));
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);

		int indexMale = labelNames.indexOf("Male");
		int indexFemale = labelNames.indexOf("Female");
		
		gnb.setLabelClassCategory("Gender");

		/*
		 Gender height (feet) weight (lbs) foot size(inches) 
		 male 6 180 12 
		 male 5.92 (5'11") 190 11 
		 male 5.58 (5'7") 170 12 
		 male 5.92 (5'11") 165 10 
		 female 5 100 6 
		 female 5.5 (5'6") 150 8 
		 female 5.42 (5'5") 130 7 
		 female 5.75 (5'9") 150 9 
		 */

		float[] dataToLoad = new float[] {6f, 180f, 12f};
		gnb.fit(dataToLoad, indexMale );
		
		dataToLoad = new float[] {5.92f,190f,11f};
		gnb.fit(dataToLoad, indexMale );
		
		dataToLoad = new float[] {5.58f,170f,12f};
		gnb.fit(dataToLoad, indexMale );
		
		dataToLoad = new float[] {5.92f,165f,10f};
		gnb.fit(dataToLoad, indexMale );

		dataToLoad = new float[] {5f,100f,6f};
		gnb.fit(dataToLoad, indexFemale );
		
		dataToLoad = new float[] {5.5f,150f,8f};
		gnb.fit(dataToLoad, indexFemale );
		
		dataToLoad = new float[] {5.42f,130f,7f};
		gnb.fit(dataToLoad, indexFemale );
		
		dataToLoad = new float[] {5.75f,150f,9f};
		gnb.fit(dataToLoad, indexFemale );
		
		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();

		dataToLoad = new float[] {6f,130f,8f};
		double[] results = gnb.getProbabilityScores_TestingSet(dataToLoad);
		
		assertEquals(6.1970717E-9, results[0], 0.000001);
		assertEquals(5.37791E-4, results[1], 0.00001);

		System.out.println("*** TEST COMPLETE ***");
	}
	
	@Test
	public void checkProp_NDArray_TwoFeatures() {
		//log.logln_withClassName(Logger.lF,"");
		log.logln_withClassName(Logger.lD,"");

		NDArray featuresTrain = NDArray.newArray(8,2);
		NDArray featuresTest = NDArray.newArray(1,2);
		double[] labels;
		
		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Two Samples - NDArray");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs)");


		/*
		 * male = 0.0
		 * female = 1.0
		 Gender height (feet) weight (lbs) 
		 male 6 180 
		 male 5.92 (5'11") 190 
		 male 5.58 (5'7") 170 
		 male 5.92 (5'11") 165 
		 female 5 100 
		 female 5.5 (5'6") 150 
		 female 5.42 (5'5") 130 
		 female 5.75 (5'9") 150 
		 */
		featuresTrain.set(0,0,6.0);
		featuresTrain.set(0,1,180.0);
		
		featuresTrain.set(1,0,5.92);
		featuresTrain.set(1,1,190.0);
		
		featuresTrain.set(2,0,5.58);
		featuresTrain.set(2,1,170.0);
		
		featuresTrain.set(3,0,5.92);
		featuresTrain.set(3,1,165.0);
		
		featuresTrain.set(4,0,5.5);
		featuresTrain.set(4,1,100.0);
		
		featuresTrain.set(5,0,5.5);
		featuresTrain.set(5,1,150.0);
		
		featuresTrain.set(6,0,5.42);
		featuresTrain.set(6,1,130.0);
		
		featuresTrain.set(7,0,5.75);
		featuresTrain.set(7,1,150.0);
		
		labels = new double[] {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
	
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_IGNORE);

		try {
		gnb.fit(featuresTrain, labels);
		} catch (Exception e) {
			e.printStackTrace();
		}

		
		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();
		
		double[] results = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		
		assertEquals(4.726183760794811E-6, results[0], 0);
		assertEquals(1.4375489263329655E-4, results[1], 0);
		
		results = null;
		featuresTest.set(0,0,6.0);
		featuresTest.set(0,1,130.0);
		
		results = gnb.getProbabilityScores_TestingSet(featuresTest);
		assertEquals(4.726183760794811E-6, results[0], 0);
		assertEquals(1.4375489263329655E-4, results[1], 0);

		double dResult = gnb.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		assertEquals(1.0, dResult, 0.0);
		double[] result = gnb.predict(featuresTest);
		assertEquals(1.0, result[0],0.0);
		double accuracy = gnb.getAccuracyOfPredictedResults(new double[]{1.0}, result);
		assertEquals (1.0, accuracy, 0.0);
		
		System.out.println("*** TEST COMPLETE ***");

	}	
	
	@Test
	public void checkProb_NDArray_TwoFeatures_TwoBatches() {
		//log.logln_withClassName(Logger.lF,"");
		log.logln_withClassName(Logger.lD,"");

		NDArray featuresTrain1 = NDArray.newArray(4,2);
		NDArray featuresTrain2 = NDArray.newArray(4,2);
		NDArray featuresTest = NDArray.newArray(1,2);

		double[] labelsTrain1;
		double[] labelsTrain2;
		
		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check NDArray - Two fit batches");


		/*
		 * male = 0.0
		 * female = 1.0
		 Gender height (feet) weight (lbs) 
		 male 6 180 
		 male 5.92 (5'11") 190 
		 male 5.58 (5'7") 170 
		 male 5.92 (5'11") 165 
		 female 5 100 
		 female 5.5 (5'6") 150 
		 female 5.42 (5'5") 130 
		 female 5.75 (5'9") 150 
		 */
		featuresTrain1.set(0,0,6.0);
		featuresTrain1.set(0,1,180.0);
		
		featuresTrain1.set(1,0,5.92);
		featuresTrain1.set(1,1,190.0);
		
		featuresTrain1.set(2,0,5.58);
		featuresTrain1.set(2,1,170.0);
		
		featuresTrain1.set(3,0,5.92);
		featuresTrain1.set(3,1,165.0);
		
		featuresTrain2.set(0,0,5.5);
		featuresTrain2.set(0,1,100.0);
		
		featuresTrain2.set(1,0,5.5);
		featuresTrain2.set(1,1,150.0);
		
		featuresTrain2.set(2,0,5.42);
		featuresTrain2.set(2,1,130.0);
		
		featuresTrain2.set(3,0,5.75);
		featuresTrain2.set(3,1,150.0);
		
		labelsTrain1 = new double[] {0.0, 0.0, 0.0, 0.0};
		labelsTrain2 = new double[] {1.0, 1.0, 1.0, 1.0};
	
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_IGNORE);

		try {
		gnb.fit(featuresTrain1, labelsTrain1);
		} catch (Exception e) {
			e.printStackTrace();
		}

		
		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();
		
		try {
		gnb.fit(featuresTrain2, labelsTrain2);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();
		
		double[] scores = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		
		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);
		
		scores = null;
		featuresTest.set(0,0,6.0);
		featuresTest.set(0,1,130.0);
		
		scores = gnb.getProbabilityScores_TestingSet(featuresTest);
		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);

		double dResult = gnb.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		assertEquals(1.0, dResult, 0.0);
		double[] result = gnb.predict(featuresTest);
		assertEquals(1.0, result[0],0.0);
		double accuracy = gnb.getAccuracyOfPredictedResults(new double[]{1.0}, result);
		assertEquals (1.0, accuracy, 0.0);
		
		System.out.println("*** TEST COMPLETE ***");

	}
	@Test
	public void checkProp_NDArray_TwoSamples_AssignedNames() {
		log.logln_withClassName(Logger.lF,"");

		NDArray featuresTrain = NDArray.newArray(8,2);
		NDArray featuresTest = NDArray.newArray(1,2);
		double[] labelsTrain;
		
		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Two Samples - NDArray");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs)");


		/*
		 * male = 0.0
		 * female = 1.0
		 Gender height (feet) weight (lbs) 
		 male 6 180 
		 male 5.92 (5'11") 190 
		 male 5.58 (5'7") 170 
		 male 5.92 (5'11") 165 
		 female 5 100 
		 female 5.5 (5'6") 150 
		 female 5.42 (5'5") 130 
		 female 5.75 (5'9") 150 
		 */
		featuresTrain.set(0,0,6.0);
		featuresTrain.set(0,1,180.0);
		
		featuresTrain.set(1,0,5.92);
		featuresTrain.set(1,1,190.0);
		
		featuresTrain.set(2,0,5.58);
		featuresTrain.set(2,1,170.0);
		
		featuresTrain.set(3,0,5.92);
		featuresTrain.set(3,1,165.0);
		
		featuresTrain.set(4,0,5.5);
		featuresTrain.set(4,1,100.0);
		
		featuresTrain.set(5,0,5.5);
		featuresTrain.set(5,1,150.0);
		
		featuresTrain.set(6,0,5.42);
		featuresTrain.set(6,1,130.0);
		
		featuresTrain.set(7,0,5.75);
		featuresTrain.set(7,1,150.0);
		
		labelsTrain = new double[] {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
		List<String> featureNames = new ArrayList<String>(Arrays.asList("Height"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male","Female"));
		// 0 = Male
		// 1 = Female
		
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);
		gnb.setLabelClassCategory("Gender");

		//GaussianNB gnb =  new GaussianNB(GaussianNB.EMPTY_SAMPLES_IGNORE);

		//		gnb.fitSample(featuresTrain, classLabelsTrain, labelList);
		try {
		gnb.fit(featuresTrain, labelsTrain);
		} catch (Exception e) {
			e.printStackTrace();
		}

		
		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();
		
		log.logln("Use method:\tpredictUsingFeatureNames (uses ArrayList<Float>)");
		double[] scores = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		
		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);
		
		scores = null;
		featuresTest.set(0,0,6.0);
		featuresTest.set(0,1,130.0);
		
		log.logln("\nUse method:\tpredictClassResults (uses NDArray)");
		scores = gnb.getProbabilityScores_TestingSet(featuresTest);
		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);

		log.logln("\nUse method:\tpredict (uses ArrayList<Float>)");
		double result = gnb.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		assertEquals(1.0, result, 0.0);
		System.out.println("Result: " + result);
		System.out.println();
		System.out.println("*** TEST COMPLETE ***");

	}	

	@Test
	public void checkMeanVar(){
		log.logln_withClassName(Logger.lF,"");
		
		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Mean & Variance (Sample)");

		NDArray featuresTrain = NDArray.newArray(6,1);
		//http://www.wikihow.com/Calculate-Variance
//17 + 15 + 23 + 7 + 9 + 13
		featuresTrain.set(0,0,17);
		featuresTrain.set(1,0,15);
		featuresTrain.set(2,0,23);
		featuresTrain.set(3,0,7);
		featuresTrain.set(4,0,9);
		featuresTrain.set(5,0,13);
		
		double[] labels = new double[] {1,1,1,1,1,1};

		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW);

		//		gnb.fitSample(featuresTrain, classLabelsTrain, labelList);
		try {
		gnb.fit(featuresTrain, labels);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printMeanVar();
		
		assertEquals(14.0, gnb.getMean(0, 0), 0.0);
		assertEquals(33.2, gnb.getVar(0, 0), 0.0);
		
		
		System.out.println("*** TEST COMPLETE ***");

	}
	@Test
	public void checkSKLearnValues() {
		log.logln_withClassName(Logger.lF,"");

		NDArray featuresTrain = NDArray.newArray(6,2);
		NDArray featuresTest = NDArray.newArray(2,2);
		NDArray sampleValues = NDArray.newArray(1,2);
		double[] labels;

		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check SKLearn Learn values");


		/*
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> Y = np.array([1, 1, 1, 2, 2, 2])
>>> from sklearn.naive_bayes import GaussianNB
>>> clf = GaussianNB()
>>> clf.fit(X, Y)
GaussianNB()
>>> print(clf.predict([[-0.8, -1]]))
[1]
		 */
		featuresTrain.set(0,0,-1);
		featuresTrain.set(0,1,-1);
		
		featuresTrain.set(1,0,-2);
		featuresTrain.set(1,1,-1);
		
		featuresTrain.set(2,0,-3);
		featuresTrain.set(2,1,-2);
		
		featuresTrain.set(3,0,1);
		featuresTrain.set(3,1,1);
		
		featuresTrain.set(4,0,2);
		featuresTrain.set(4,1,1);
		
		featuresTrain.set(5,0,3);
		featuresTrain.set(5,1,2);
		
		labels = new double[] {1,1,1,2,2,2};
//		List<String> labelList = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)"));
	
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(
				GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW);

		//		gnb.fitSample(featuresTrain, classLabelsTrain, labelList);
		try {
		gnb.fit(featuresTrain, labels);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();
		featuresTest.set(0,0,-0.8);
		featuresTest.set(0,1,-1);
		featuresTest.set(1,0,1);
		featuresTest.set(1,1,2);

		double[] result = gnb.predict(featuresTest);
		
		double accuracy = gnb.getAccuracyOfPredictedResults(new double[] {1.0, 2.0}, result);
		System.out.println("Samples: " + featuresTest);
		System.out.println("Results: ");
		for (double dVal : result) {
			System.out.println (dVal);
		}
		System.out.println("Accuracy: " + accuracy);

		System.out.println("Scores: " );
		sampleValues.set(0,0,-0.8);
		sampleValues.set(0,1,-1);
		System.out.println("Values: " + sampleValues);
		double[] scores = gnb.getProbabilityScores_TestingSet(sampleValues);
		System.out.println(ArrayUtils.printArray(scores));
		
		sampleValues.set(0,0,1);
		sampleValues.set(0,1,2);
		System.out.println("Values: " + sampleValues);
		
		scores = gnb.getProbabilityScores_TestingSet(sampleValues);
		System.out.println(ArrayUtils.printArray(scores));
		
		System.out.println();
		
		System.out.println("*** TEST COMPLETE ***");
		assertEquals(1.0, result[0], 0.0);
		assertEquals(2.0, result[1], 0.0);
		assertEquals(1.0, accuracy, 0.0);

	}
	
	@Test
	public void checkGNBSerialization() {
		log.logln_withClassName(Logger.lF,"");

		NDArray featuresTrain = NDArray.newArray(8,2);
		NDArray featuresTest = NDArray.newArray(1,2);
		double[] labelsTrain;
		
		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Two Samples - NDArray");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs)");


		/*
		 * male = 0.0
		 * female = 1.0
		 Gender height (feet) weight (lbs) 
		 male 6 180 
		 male 5.92 (5'11") 190 
		 male 5.58 (5'7") 170 
		 male 5.92 (5'11") 165 
		 female 5 100 
		 female 5.5 (5'6") 150 
		 female 5.42 (5'5") 130 
		 female 5.75 (5'9") 150 
		 */
		featuresTrain.set(0,0,6.0);
		featuresTrain.set(0,1,180.0);
		
		featuresTrain.set(1,0,5.92);
		featuresTrain.set(1,1,190.0);
		
		featuresTrain.set(2,0,5.58);
		featuresTrain.set(2,1,170.0);
		
		featuresTrain.set(3,0,5.92);
		featuresTrain.set(3,1,165.0);
		
		featuresTrain.set(4,0,5.5);
		featuresTrain.set(4,1,100.0);
		
		featuresTrain.set(5,0,5.5);
		featuresTrain.set(5,1,150.0);
		
		featuresTrain.set(6,0,5.42);
		featuresTrain.set(6,1,130.0);
		
		featuresTrain.set(7,0,5.75);
		featuresTrain.set(7,1,150.0);
		
		labelsTrain = new double[] {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
		List<String> featureNames = new ArrayList<String>(Arrays.asList("Height"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male","Female"));
		// 0 = Male
		// 1 = Female
		
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);
		gnb.setLabelClassCategory("Gender");

		//GaussianNB gnb =  new GaussianNB(GaussianNB.EMPTY_SAMPLES_IGNORE);

		//		gnb.fitSample(featuresTrain, classLabelsTrain, labelList);
		try {
		gnb.fit(featuresTrain, labelsTrain);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		AccessSerializedObject.writeClassGNB(gnb, "testGNBserialization.ser");
		
		gnb = null;
		GaussianNaiveBayesClassifier gnb2 =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);
		
		gnb2 = AccessSerializedObject.readClassGNB("testGNBserialization.ser");
		
		gnb2.printFeaturesAndLabels();
		gnb2.printMeanVar();
		
		log.logln("Use method:\tpredictUsingFeatureNames (uses ArrayList<Float>)");
		double[] scores = gnb2.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		
		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);
		
		scores = null;
		featuresTest.set(0,0,6.0);
		featuresTest.set(0,1,130.0);
		
		log.logln("\nUse method:\tpredictClassResults (uses NDArray)");
		scores = gnb2.getProbabilityScores_TestingSet(featuresTest);
		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);

		log.logln("\nUse method:\tpredict (uses ArrayList<Float>)");
		double result = gnb2.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f)));
		assertEquals(1.0, result, 0.0);
		System.out.println("Result: " + result);
		System.out.println();
		System.out.println("*** TEST COMPLETE ***");

	}	
	
	@Test
	public void checkProbability_ThreeFeatures_SingleTestValue() {
		log.logln_withClassName(Logger.lD,"");

		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Three Samples");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs, 8\" shoe)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs, 8\" shoe)");

		List<String> featureNames = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)","Ft(in)"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male","Female"));
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);

		int indexMale = labelNames.indexOf("Male");
		int indexFemale = labelNames.indexOf("Female");
		
		gnb.setLabelClassCategory("Gender");

		/*
		 Gender height (feet) weight (lbs) foot size(inches) 
		 male 6 180 12 
		 male 5.92 (5'11") 190 11 
		 male 5.58 (5'7") 170 12 
		 male 5.92 (5'11") 165 10 
		 female 5 100 6 
		 female 5.5 (5'6") 150 8 
		 female 5.42 (5'5") 130 7 
		 female 5.75 (5'9") 150 9 
		 */
		gnb.fit(new ArrayList<Float>(Arrays.asList(6f,180f,12f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,190f,11f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.58f,170f,12f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,165f,10f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5f,100f,6f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.5f,150f,8f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.42f,130f,7f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.75f,150f,9f)), indexFemale);

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();

		double[] results = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f,130f,8f)));
		
		assertEquals(6.1970717E-9, results[0], 0.000001);
		assertEquals(5.37791E-4, results[1], 0.00001);

		double label = gnb.predict_OneTestValue(0, 5.75f);
		System.out.println("gender: " + label + ", using feat index: " + 0 + ", height: " + 5.75f);
		assertEquals(0, label, 0);
		
		label = gnb.predict_TestingSet(new float[]{5.75f,150f,9f});
		System.out.println("gender: " + label + ", " + ArrayUtils.printArray(new float[]{5.75f,150f,9f}));

		assertEquals(1, label, 0);
		
		System.out.println("*** TEST COMPLETE ***");
	}

	@Test
	public void checkProbability_ThreeFeatures_EmptyTestValue_AllowEmptyValues() {
		// Logger.setLoggerLevel(Logger.LOG_OFF);
		//Logger.setLoggerLevel(Logger.LOG_FINE);
		// Logger.setLoggerLevel(Logger.LOG_INFO);
		Logger.setLoggerLevel(Logger.LOG_DEBUG);
		log.logln_withClassName(Logger.lF, "Testing");

		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Three Samples");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs, 8\" shoe)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs, 8\" shoe)");

		List<String> featureNames = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)","Ft(in)"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male","Female"));
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, labelNames, featureNames);

		int indexMale = labelNames.indexOf("Male");
		int indexFemale = labelNames.indexOf("Female");
		
		gnb.setLabelClassCategory("Gender");

		/*
		 Gender height (feet) weight (lbs) foot size(inches) 
		 male 6 180 12 
		 male 5.92 (5'11") 190 11 
		 male 5.58 (5'7") 170 12 
		 male 5.92 (5'11") 165 10 
		 female 5 100 6 
		 female 5.5 (5'6") 150 8 
		 female 5.42 (5'5") 130 7 
		 female 5.75 (5'9") 150 9 
		 */
		gnb.fit(new ArrayList<Float>(Arrays.asList(0f,180f,12f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,190f,11f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.58f,170f,12f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,165f,10f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5f,100f,6f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.5f,150f,8f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.42f,130f,7f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.75f,150f,9f)), indexFemale);

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();

		double[] results = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(0f,130f,8f)));
		
		assertEquals(0, results[0], 0.000001);
		assertEquals(0, results[1], 0.00001);


		double label = gnb.predict_TestingSet(new float[]{0,150f,9f});
		System.out.println("gender: " + label + ", " + ArrayUtils.printArray(new float[]{0,150f,9f}));

		assertEquals(0, label, 0);
		
		System.out.println("*** TEST COMPLETE ***");
	}

	@Test
	public void checkProbability_ThreeFeatures_EmptyTestValue_NotAllowEmptyValues() {
		// Logger.setLoggerLevel(Logger.LOG_OFF);
		//Logger.setLoggerLevel(Logger.LOG_FINE);
		// Logger.setLoggerLevel(Logger.LOG_INFO);
		Logger.setLoggerLevel(Logger.LOG_DEBUG);
		log.logln_withClassName(Logger.lF, "Testing...");

		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Three Samples");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs, 8\" shoe)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs, 8\" shoe)");

		List<String> featureNames = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)","Ft(in)"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male","Female"));
		GaussianNaiveBayesClassifier gnb =  new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_IGNORE, labelNames, featureNames );

		int indexMale = labelNames.indexOf("Male");
		int indexFemale = labelNames.indexOf("Female");
		
		gnb.setLabelClassCategory("Gender");

		/*
		 Gender height (feet) weight (lbs) foot size(inches) 
		 male 6 180 12 
		 male 5.92 (5'11") 190 11 
		 male 5.58 (5'7") 170 12 
		 male 5.92 (5'11") 165 10 
		 female 5 100 6 
		 female 5.5 (5'6") 150 8 
		 female 5.42 (5'5") 130 7 
		 female 5.75 (5'9") 150 9 
		 */
		gnb.fit(new ArrayList<Float>(Arrays.asList(0f,180f,12f)), indexMale);
		gnb.fit(new float[]{0f,180f,12f}, indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,190f,11f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.58f,170f,12f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,165f,10f)), indexMale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5f,100f,6f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.5f,150f,8f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.42f,130f,7f)), indexFemale);
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.75f,150f,9f)), indexFemale);

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();

		double[] results = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(0f,130f,8f)));
		
		assertEquals(0.000000000061715605, results[0], 0.000001);
		assertEquals(0.0023104010615497828, results[1], 0.00001);


		double label = gnb.predict_TestingSet(new float[]{0,150f,9f});
		System.out.println("gender: " + label + ", " + ArrayUtils.printArray(new float[]{0,150f,9f}));

		assertEquals(1, label, 0);
		
		System.out.println("*** TEST COMPLETE ***");
	}


}
