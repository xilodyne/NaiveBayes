package xilodyne.machinelearning.classifier.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.TreeMap;

import mikera.arrayz.NDArray;

import org.junit.Test;

import xilodyne.machinelearning.classifier.bayes.GaussianNaiveBayesClassifier;
import xilodyne.machinelearning.classifier.bayes.NaiveBayesClassifier;
import xilodyne.util.G;
import xilodyne.util.Logger;
import static org.junit.Assert.assertEquals;

public class Test_NaiveBayes {

	private Logger log = new Logger();
	private int nextNumber = 0;

	public Test_NaiveBayes() {
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		G.setLoggerLevel(G.LOG_DEBUG);

	}

	@Test
	public void checkProbabilityOneLabel() {
		// setup
		System.out.println();
		System.out.println("*** TEST *** Check Probability One Label");
		System.out.println("Testing:  P(Male|Drew)");
		System.out.println("Testing:  P(Female|Drew)");
		System.out.println();

		//String[] featureCategories = new String[] { "Name" };
		//String[] labels = new String[] { "Male", "Female" };

		NaiveBayesClassifier nb = new NaiveBayesClassifier(NaiveBayesClassifier.EMPTY_SAMPLES_IGNORE);

		// create temp storage to get unique id for each string
		TreeMap<String, Double> stringUniqueID = new TreeMap<String, Double>();
		stringUniqueID.put("Drew", (double) this.getNextID());
		stringUniqueID.put("Claudia", (double) this.getNextID());
		stringUniqueID.put("Alberto", (double) this.getNextID());
		stringUniqueID.put("Karin", (double) this.getNextID());
		stringUniqueID.put("Nina", (double) this.getNextID());
		stringUniqueID.put("Sergio", (double) this.getNextID());

		nb.fit(0, stringUniqueID.get("Drew"), (double) 0);
		nb.fit(0, stringUniqueID.get("Claudia"), (double) 1);
		nb.fit(0, stringUniqueID.get("Drew"), (double) 1);
		nb.fit(0, stringUniqueID.get("Drew"), (double) 1);
		nb.fit(0, stringUniqueID.get("Alberto"), (double) 0);
		nb.fit(0, stringUniqueID.get("Karin"), (double) 1);
		nb.fit(0, stringUniqueID.get("Nina"), (double) 1);
		nb.fit(0, stringUniqueID.get("Sergio"), (double) 0);

		nb.printFeaturesAndLabels();

		double result = 0;
		double id = stringUniqueID.get("Drew");
		result = nb.getProbabilty_OneFeature(0, 0, (float) id);
		assertEquals(0.125, result, 0);

		result = nb.getProbabilty_OneFeature(0, 1, (float) id);

		assertEquals(0.25, result, 0);
		System.out.println("*** TEST COMPLETE ***");
	}

	@Test
	public void checkProbabilityFourFeatures() {
		System.out.println();
		System.out.println("*** TEST *** Check Probability Four Features");
		System.out.println("Testing:  P(Male|Drew,>170cm,Blue Eyes, Long Hair)");
		System.out.println("Testing:  P(Female|Drew,>170cm,Blue Eyes, Long Hair)");
		String[] featureCategories = new String[] { "Name", ">170cm", "Eye", "Hair" };
		String[] labels = new String[] { "Male", "Female" };
		List<String> featureNames = Arrays.asList(featureCategories);
		List<String> labelNames = Arrays.asList(labels);

		NaiveBayesClassifier nb = new NaiveBayesClassifier(NaiveBayesClassifier.EMPTY_SAMPLES_IGNORE, featureNames,
				labelNames);

		// create temp storage to get unique id for each string
		TreeMap<String, Double> stringUniqueID = new TreeMap<String, Double>();
		stringUniqueID.put("Drew", (double) this.getNextID());
		stringUniqueID.put("Claudia", (double) this.getNextID());
		stringUniqueID.put("Alberto", (double) this.getNextID());
		stringUniqueID.put("Karin", (double) this.getNextID());
		stringUniqueID.put("Nina", (double) this.getNextID());
		stringUniqueID.put("Sergio", (double) this.getNextID());
		stringUniqueID.put("No", (double) this.getNextID());
		stringUniqueID.put("Yes", (double) this.getNextID());
		stringUniqueID.put("Blue", (double) this.getNextID());
		stringUniqueID.put("Brown", (double) this.getNextID());
		stringUniqueID.put("Short", (double) this.getNextID());
		stringUniqueID.put("Long", (double) this.getNextID());

		nb.fit(0, stringUniqueID.get("Drew"), (double) 0);
		nb.fit(1, stringUniqueID.get("No"), (double) 0);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 0);
		nb.fit(3, stringUniqueID.get("Short"), (double) 0);

		nb.fit(0, stringUniqueID.get("Claudia"), (double) 1);
		nb.fit(1, stringUniqueID.get("Yes"), (double) 1);
		nb.fit(2, stringUniqueID.get("Brown"), (double) 1);
		nb.fit(3, stringUniqueID.get("Long"), (double) 1);

		nb.fit(0, stringUniqueID.get("Drew"), (double) 1);
		nb.fit(1, stringUniqueID.get("No"), (double) 1);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 1);
		nb.fit(3, stringUniqueID.get("Long"), (double) 1);

		nb.fit(0, stringUniqueID.get("Drew"), (double) 1);
		nb.fit(1, stringUniqueID.get("No"), (double) 1);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 1);
		nb.fit(3, stringUniqueID.get("Long"), (double) 1);

		nb.fit(0, stringUniqueID.get("Alberto"), (double) 0);
		nb.fit(1, stringUniqueID.get("Yes"), (double) 0);
		nb.fit(2, stringUniqueID.get("Brown"), (double) 0);
		nb.fit(3, stringUniqueID.get("Short"), (double) 0);

		nb.fit(0, stringUniqueID.get("Karin"), (double) 1);
		nb.fit(1, stringUniqueID.get("No"), (double) 1);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 1);
		nb.fit(3, stringUniqueID.get("Long"), (double) 1);

		nb.fit(0, stringUniqueID.get("Nina"), (double) 1);
		nb.fit(1, stringUniqueID.get("Yes"), (double) 1);
		nb.fit(2, stringUniqueID.get("Brown"), (double) 1);
		nb.fit(3, stringUniqueID.get("Short"), (double) 1);

		nb.fit(0, stringUniqueID.get("Sergio"), (double) 0);
		nb.fit(1, stringUniqueID.get("Yes"), (double) 0);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 0);
		nb.fit(3, stringUniqueID.get("Long"), (double) 0);

		/*
		 * ("Drew","No","Blue","Short"), "Male"
		 * ("Claudia","Yes","Brown","Long"), "Female"
		 * ("Drew","No","Blue","Long"), "Female" ("Drew","No","Blue","Long"),
		 * "Female" ("Alberto","Yes","Brown","Short"), "Male"
		 * ("Karin","No","Blue","Long"), "Female"
		 * ("Nina","Yes","Brown","Short"), "Female"
		 * ("Sergio","Yes","Blue","Long"), "Male"
		 */
		nb.printFeaturesAndLabels();
		//double predictedLabel = 0;
		List<Float> testingData = new ArrayList<Float>();
		double val = stringUniqueID.get("Drew");
		testingData.add((float) val);

		val = stringUniqueID.get("Yes");
		testingData.add((float) val);

		val = stringUniqueID.get("Blue");
		testingData.add((float) val);

		val = stringUniqueID.get("Long");
		testingData.add((float) val);

		double predictedLabel = nb.predict_TestingSet(testingData);
		System.out.println("Predicted Label: " + predictedLabel);

		// assertEquals("FEMALE", predictedLabel, 0);

		double[] results = nb.getProbabilityScores_TestingSet(testingData);
		assertEquals(0.019, results[0], 0.001);
		assertEquals(0.048, results[1], 0.001);

		System.out.println("*** TEST COMPLETE ***");
	}

	@Test
	public void checkProbabilityOneFeatureAllClasses() {
		System.out.println();
		System.out.println("*** TEST *** Check Probability One Feature All Classes");
		System.out.println("Testing:  P(Male|Drew)");
		System.out.println("Testing:  P(Female|Drew)");
		System.out.println("Testing:  P(Male|Long hair)");
		System.out.println("Testing:  P(Female|Long hair)");
		System.out.println();

		//String[] featureCategories = new String[] { "Name", ">170cm", "Eye", "Hair" };
		//String[] labels = new String[] { "Male", "Female" };

		NaiveBayesClassifier nb = new NaiveBayesClassifier(NaiveBayesClassifier.EMPTY_SAMPLES_IGNORE);

		// create temp storage to get unique id for each string
		TreeMap<String, Double> stringUniqueID = new TreeMap<String, Double>();
		stringUniqueID.put("Drew", (double) this.getNextID());
		stringUniqueID.put("Claudia", (double) this.getNextID());
		stringUniqueID.put("Alberto", (double) this.getNextID());
		stringUniqueID.put("Karin", (double) this.getNextID());
		stringUniqueID.put("Nina", (double) this.getNextID());
		stringUniqueID.put("Sergio", (double) this.getNextID());
		stringUniqueID.put("No", (double) this.getNextID());
		stringUniqueID.put("Yes", (double) this.getNextID());
		stringUniqueID.put("Blue", (double) this.getNextID());
		stringUniqueID.put("Brown", (double) this.getNextID());
		stringUniqueID.put("Short", (double) this.getNextID());
		stringUniqueID.put("Long", (double) this.getNextID());

		nb.fit(0, stringUniqueID.get("Drew"), (double) 0);
		nb.fit(1, stringUniqueID.get("No"), (double) 0);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 0);
		nb.fit(3, stringUniqueID.get("Short"), (double) 0);

		nb.fit(0, stringUniqueID.get("Claudia"), (double) 1);
		nb.fit(1, stringUniqueID.get("Yes"), (double) 1);
		nb.fit(2, stringUniqueID.get("Brown"), (double) 1);
		nb.fit(3, stringUniqueID.get("Long"), (double) 1);

		nb.fit(0, stringUniqueID.get("Drew"), (double) 1);
		nb.fit(1, stringUniqueID.get("No"), (double) 1);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 1);
		nb.fit(3, stringUniqueID.get("Long"), (double) 1);

		nb.fit(0, stringUniqueID.get("Drew"), (double) 1);
		nb.fit(1, stringUniqueID.get("No"), (double) 1);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 1);
		nb.fit(3, stringUniqueID.get("Long"), (double) 1);

		nb.fit(0, stringUniqueID.get("Alberto"), (double) 0);
		nb.fit(1, stringUniqueID.get("Yes"), (double) 0);
		nb.fit(2, stringUniqueID.get("Brown"), (double) 0);
		nb.fit(3, stringUniqueID.get("Short"), (double) 0);

		nb.fit(0, stringUniqueID.get("Karin"), (double) 1);
		nb.fit(1, stringUniqueID.get("No"), (double) 1);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 1);
		nb.fit(3, stringUniqueID.get("Long"), (double) 1);

		nb.fit(0, stringUniqueID.get("Nina"), (double) 1);
		nb.fit(1, stringUniqueID.get("Yes"), (double) 1);
		nb.fit(2, stringUniqueID.get("Brown"), (double) 1);
		nb.fit(3, stringUniqueID.get("Short"), (double) 1);

		nb.fit(0, stringUniqueID.get("Sergio"), (double) 0);
		nb.fit(1, stringUniqueID.get("Yes"), (double) 0);
		nb.fit(2, stringUniqueID.get("Blue"), (double) 0);
		nb.fit(3, stringUniqueID.get("Long"), (double) 0);

		/*
		 * ("Drew","No","Blue","Short"), "Male"
		 * ("Claudia","Yes","Brown","Long"), "Female"
		 * ("Drew","No","Blue","Long"), "Female" ("Drew","No","Blue","Long"),
		 * "Female" ("Alberto","Yes","Brown","Short"), "Male"
		 * ("Karin","No","Blue","Long"), "Female"
		 * ("Nina","Yes","Brown","Short"), "Female"
		 * ("Sergio","Yes","Blue","Long"), "Male"
		 */

		nb.printFeaturesAndLabels();

		double result = 0;
		double val = stringUniqueID.get("Drew");
		result = nb.getProbabilty_OneFeature(0, 0, (float) val);
		assertEquals(0.125, result, 0);

		val = stringUniqueID.get("Drew");
		result = nb.getProbabilty_OneFeature(0, 1, (float) val);
		assertEquals(0.25, result, 0);

		val = stringUniqueID.get("Long");
		result = nb.getProbabilty_OneFeature(3, 0, (float) val);
		assertEquals(0.125, result, 0);

		val = stringUniqueID.get("Long");
		result = nb.getProbabilty_OneFeature(3, 1, (float) val);
		assertEquals(0.5, result, 0);

		System.out.println("*** TEST COMPLETE ***");

	}

	@Test
	public void checkProp_NDArray_TwoFeatures() {
		// log.logln_withClassName(G.lF,"");
		log.logln_withClassName(G.lD, "");

		NDArray featuresTrain = NDArray.newArray(8, 2);
		NDArray featuresTest = NDArray.newArray(1, 2);
		double[] labels;

		System.out.println();
		System.out.println();
		System.out.println("*** TEST *** Check Probability Two Samples - NDArray");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs)");

		/*
		 * male = 0.0 female = 1.0 Gender height (feet) weight (lbs) male 6 180
		 * male 5.92 (5'11") 190 male 5.58 (5'7") 170 male 5.92 (5'11") 165
		 * female 5 100 female 5.5 (5'6") 150 female 5.42 (5'5") 130 female 5.75
		 * (5'9") 150
		 */
		featuresTrain.set(0, 0, 6.0);
		featuresTrain.set(0, 1, 180.0);

		featuresTrain.set(1, 0, 5.92);
		featuresTrain.set(1, 1, 190.0);

		featuresTrain.set(2, 0, 5.58);
		featuresTrain.set(2, 1, 170.0);

		featuresTrain.set(3, 0, 5.92);
		featuresTrain.set(3, 1, 165.0);

		featuresTrain.set(4, 0, 5.5);
		featuresTrain.set(4, 1, 100.0);

		featuresTrain.set(5, 0, 5.5);
		featuresTrain.set(5, 1, 150.0);

		featuresTrain.set(6, 0, 5.42);
		featuresTrain.set(6, 1, 130.0);

		featuresTrain.set(7, 0, 5.75);
		featuresTrain.set(7, 1, 150.0);

		labels = new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 };

		GaussianNaiveBayesClassifier gnb = new GaussianNaiveBayesClassifier(
				GaussianNaiveBayesClassifier.EMPTY_SAMPLES_IGNORE);

		try {
			gnb.fit(featuresTrain, labels);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();

		double[] results = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f, 130f)));

		assertEquals(4.726183760794811E-6, results[0], 0);
		assertEquals(1.4375489263329655E-4, results[1], 0);

		results = null;
		featuresTest.set(0, 0, 6.0);
		featuresTest.set(0, 1, 130.0);

		results = gnb.getProbabilityScores_TestingSet(featuresTest);
		assertEquals(4.726183760794811E-6, results[0], 0);
		assertEquals(1.4375489263329655E-4, results[1], 0);

		double dResult = gnb.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f, 130f)));
		assertEquals(1.0, dResult, 0.0);
		double[] result = gnb.predict(featuresTest);
		assertEquals(1.0, result[0], 0.0);
		double accuracy = gnb.getAccuracyOfPredictedResults(new double[] { 1.0 }, result);
		assertEquals(1.0, accuracy, 0.0);

		System.out.println("*** TEST COMPLETE ***");

	}

	@Test
	public void checkProb_NDArray_TwoFeatures_TwoBatches() {
		// log.logln_withClassName(G.lF,"");
		log.logln_withClassName(G.lD, "");

		NDArray featuresTrain1 = NDArray.newArray(4, 2);
		NDArray featuresTrain2 = NDArray.newArray(4, 2);
		NDArray featuresTest = NDArray.newArray(1, 2);

		double[] labelsTrain1;
		double[] labelsTrain2;

		System.out.println();
		System.out.println();
		System.out.println("*** TEST *** Check NDArray - Two fit batches");

		/*
		 * male = 0.0 female = 1.0 Gender height (feet) weight (lbs) male 6 180
		 * male 5.92 (5'11") 190 male 5.58 (5'7") 170 male 5.92 (5'11") 165
		 * female 5 100 female 5.5 (5'6") 150 female 5.42 (5'5") 130 female 5.75
		 * (5'9") 150
		 */
		featuresTrain1.set(0, 0, 6.0);
		featuresTrain1.set(0, 1, 180.0);

		featuresTrain1.set(1, 0, 5.92);
		featuresTrain1.set(1, 1, 190.0);

		featuresTrain1.set(2, 0, 5.58);
		featuresTrain1.set(2, 1, 170.0);

		featuresTrain1.set(3, 0, 5.92);
		featuresTrain1.set(3, 1, 165.0);

		featuresTrain2.set(0, 0, 5.5);
		featuresTrain2.set(0, 1, 100.0);

		featuresTrain2.set(1, 0, 5.5);
		featuresTrain2.set(1, 1, 150.0);

		featuresTrain2.set(2, 0, 5.42);
		featuresTrain2.set(2, 1, 130.0);

		featuresTrain2.set(3, 0, 5.75);
		featuresTrain2.set(3, 1, 150.0);

		labelsTrain1 = new double[] { 0.0, 0.0, 0.0, 0.0 };
		labelsTrain2 = new double[] { 1.0, 1.0, 1.0, 1.0 };

		GaussianNaiveBayesClassifier gnb = new GaussianNaiveBayesClassifier(
				GaussianNaiveBayesClassifier.EMPTY_SAMPLES_IGNORE);

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

		double[] scores = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f, 130f)));

		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);

		scores = null;
		featuresTest.set(0, 0, 6.0);
		featuresTest.set(0, 1, 130.0);

		scores = gnb.getProbabilityScores_TestingSet(featuresTest);
		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);

		double dResult = gnb.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f, 130f)));
		assertEquals(1.0, dResult, 0.0);
		double[] result = gnb.predict(featuresTest);
		assertEquals(1.0, result[0], 0.0);
		double accuracy = gnb.getAccuracyOfPredictedResults(new double[] { 1.0 }, result);
		assertEquals(1.0, accuracy, 0.0);

		System.out.println("*** TEST COMPLETE ***");

	}

	@Test
	public void checkProp_NDArray_TwoSamples_AssignedNames() {
		log.logln_withClassName(G.lF, "");

		NDArray featuresTrain = NDArray.newArray(8, 2);
		NDArray featuresTest = NDArray.newArray(1, 2);
		double[] labelsTrain;

		System.out.println();
		System.out.println();
		System.out.println("*** TEST *** Check Probability Two Samples - NDArray");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs)");

		/*
		 * male = 0.0 female = 1.0 Gender height (feet) weight (lbs) male 6 180
		 * male 5.92 (5'11") 190 male 5.58 (5'7") 170 male 5.92 (5'11") 165
		 * female 5 100 female 5.5 (5'6") 150 female 5.42 (5'5") 130 female 5.75
		 * (5'9") 150
		 */
		featuresTrain.set(0, 0, 6.0);
		featuresTrain.set(0, 1, 180.0);

		featuresTrain.set(1, 0, 5.92);
		featuresTrain.set(1, 1, 190.0);

		featuresTrain.set(2, 0, 5.58);
		featuresTrain.set(2, 1, 170.0);

		featuresTrain.set(3, 0, 5.92);
		featuresTrain.set(3, 1, 165.0);

		featuresTrain.set(4, 0, 5.5);
		featuresTrain.set(4, 1, 100.0);

		featuresTrain.set(5, 0, 5.5);
		featuresTrain.set(5, 1, 150.0);

		featuresTrain.set(6, 0, 5.42);
		featuresTrain.set(6, 1, 130.0);

		featuresTrain.set(7, 0, 5.75);
		featuresTrain.set(7, 1, 150.0);

		labelsTrain = new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 };
		List<String> featureNames = new ArrayList<String>(Arrays.asList("Height"));
		List<String> labelNames = new ArrayList<String>(Arrays.asList("Male", "Female"));
		// 0 = Male
		// 1 = Female

		GaussianNaiveBayesClassifier gnb = new GaussianNaiveBayesClassifier(
				GaussianNaiveBayesClassifier.EMPTY_SAMPLES_ALLOW, featureNames, labelNames);
		gnb.setLabelClassCategory("Gender");

		// GaussianNB gnb = new GaussianNB(GaussianNB.EMPTY_SAMPLES_IGNORE);

		// gnb.fitSample(featuresTrain, classLabelsTrain, labelList);
		try {
			gnb.fit(featuresTrain, labelsTrain);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printFeaturesAndLabels();
		gnb.printMeanVar();

		log.logln("Use method:\tpredictUsingFeatureNames (uses ArrayList<Float>)");
		double[] scores = gnb.getProbabilityScores_TestingSet(new ArrayList<Float>(Arrays.asList(6f, 130f)));

		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);

		scores = null;
		featuresTest.set(0, 0, 6.0);
		featuresTest.set(0, 1, 130.0);

		log.logln("\nUse method:\tpredictClassResults (uses NDArray)");
		scores = gnb.getProbabilityScores_TestingSet(featuresTest);
		assertEquals(4.726183760794811E-6, scores[0], 0);
		assertEquals(1.4375489263329655E-4, scores[1], 0);

		log.logln("\nUse method:\tpredict (uses ArrayList<Float>)");
		double result = gnb.predict_TestingSet(new ArrayList<Float>(Arrays.asList(6f, 130f)));
		assertEquals(1.0, result, 0.0);
		System.out.println("Result: " + result);
		System.out.println();
		System.out.println("*** TEST COMPLETE ***");

	}

	private int getNextID() {
		this.nextNumber++;
		return this.nextNumber;
	}

}
