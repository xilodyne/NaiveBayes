package xilodyne.machinelearning.classifier.test;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;




import mikera.arrayz.NDArray;

import org.junit.Test;

import xilodyne.machinelearning.classifier.GaussianNB;
import xilodyne.util.G;
import xilodyne.util.Logger;


public class Test_GaussianNB {
	private Logger log = new Logger();
	
	public Test_GaussianNB() {
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);
	}
	
	@Test
	public void checkProbabilityOneLabel() {
		log.logln_withClassName(G.lF,"");	
			
		//setup
		System.out.println();
		System.out.println("*** TEST *** Check Probability One Label");
		System.out.println("Testing:  P(Male|Height)");
		System.out.println();
		
		List <String> labelList = new ArrayList<String>(Arrays.asList("Height"));
		List<String> classes = new ArrayList<String>(Arrays.asList("Male","Female"));

		GaussianNB gnb =  new GaussianNB(GaussianNB.EMPTY_SAMPLES_ALLOW, classes, labelList);
		gnb.setClassListDisplayName("Gender");
		
		int labelIndex = labelList.indexOf("Height");
		gnb.fit(labelIndex, 6f, "Male");
		gnb.fit(labelIndex, 5.92f, "Male");
		gnb.fit(labelIndex, 5.58f, "Male");
		gnb.fit(labelIndex, 5.92f, "Male");
		gnb.fit(labelIndex, 5f, "Female");
		gnb.fit(labelIndex, 5.5f, "Female");
		gnb.fit(labelIndex, 5.42f, "Female");
		gnb.fit(labelIndex, 5.75f, "Female");

		gnb.printAttributeValuesAndClasses();
		gnb.printMeanVar();
		
		float result = gnb.predictSingleLabelSingleClass(classes.indexOf("Male"), labelList.indexOf("Height"), 6.0f);
		assertEquals(0.7894416, result, 0.000001);
		
		result = gnb.predictSingleLabelSingleClass(classes.indexOf("Female"), labelList.indexOf("Height"), 6.0f);
		assertEquals(0.11172937, result, 0.000001);
		System.out.println("*** TEST COMPLETE ***");
	}

	@Test
	public void checkProbabilityThreeSamples() {
		log.logln_withClassName(G.lF,"");

		System.out.println();		
		System.out.println();
		System.out.println("*** TEST *** Check Probability Three Samples");
		System.out.println("Testing:  P(Male  |6ft, 130 lbs, 8\" shoe)");
		System.out.println("Testing:  P(Female|6ft, 130 lbs, 8\" shoe)");

		List<String> labelList = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)","Ft(in)"));
		List<String> classes = new ArrayList<String>(Arrays.asList("Male","Female"));
		GaussianNB gnb =  new GaussianNB(GaussianNB.EMPTY_SAMPLES_ALLOW, classes, labelList);

		gnb.setClassListDisplayName("Gender");

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
		gnb.fit(new ArrayList<Float>(Arrays.asList(6f,180f,12f)), "Male");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,190f,11f)), "Male");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.58f,170f,12f)), "Male");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.92f,165f,10f)), "Male");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5f,100f,6f)), "Female");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.5f,150f,8f)), "Female");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.42f,130f,7f)), "Female");
		gnb.fit(new ArrayList<Float>(Arrays.asList(5.75f,150f,9f)), "Female");

		gnb.printAttributeValuesAndClasses();
		gnb.printMeanVar();

		float[] results = gnb.getScoresFromPrediction(new ArrayList<Float>(Arrays.asList(6f,130f,8f)));
		
		assertEquals(6.1970717E-9, results[0], 0.000001);
		assertEquals(5.37791E-4, results[1], 0.00001);

		System.out.println("*** TEST COMPLETE ***");
	}
	
	@Test
	public void checkNDArray() {
		log.logln_withClassName(G.lF,"");

		NDArray featuresTrain = NDArray.newArray(8,2);
		NDArray featuresTest = NDArray.newArray(1,2);
		double[] labeled;
		
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
		
		labeled = new double[] {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
	
		GaussianNB gnb =  new GaussianNB(GaussianNB.EMPTY_SAMPLES_IGNORE);

		try {
		gnb.fit(featuresTrain, labeled);
		} catch (Exception e) {
			e.printStackTrace();
		}

		
		gnb.printAttributeValuesAndClasses();
		gnb.printMeanVar();
		
		float[] results = gnb.getScoresFromPrediction(new ArrayList<Float>(Arrays.asList(6f,130f)));
		
		assertEquals(4.726183760794811E-6, results[0], 0);
		assertEquals(1.4375489263329655E-4, results[1], 0);
		
		results = null;
		featuresTest.set(0,0,6.0);
		featuresTest.set(0,1,130.0);
		
		results = gnb.predictClassResults(featuresTest);
		assertEquals(4.726183760794811E-6, results[0], 0);
		assertEquals(1.4375489263329655E-4, results[1], 0);

		String sResult = gnb.predict(new ArrayList<Float>(Arrays.asList(6f,130f)));
		assertEquals("1.0", sResult);
		double[] result = gnb.predict(featuresTest);
		assertEquals(1.0, result[0],0.0);
		double accuracy = gnb.getAccuracyOfPredictedResults(new double[]{1.0}, result);
		assertEquals (1.0, accuracy, 0.0);
		
		System.out.println("*** TEST COMPLETE ***");

	}	
	@Test
	public void checkNDArrayWithAssignedNamedValues() {
		log.logln_withClassName(G.lF,"");

		NDArray featuresTrain = NDArray.newArray(8,2);
		NDArray featuresTest = NDArray.newArray(1,2);
		double[] labeled;
		
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
		
		labeled = new double[] {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0};
	
		GaussianNB gnb =  new GaussianNB(GaussianNB.EMPTY_SAMPLES_IGNORE);

		//		gnb.fitSample(featuresTrain, classLabelsTrain, labelList);
		try {
		gnb.fit(featuresTrain, labeled);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.setClassListDisplayName("Gender");
		gnb.setClassNames(new String[] {"Male", "Female"});
		gnb.setLabelNames(new String[] {"Height", "Weight"});
		
		gnb.printAttributeValuesAndClasses();
		gnb.printMeanVar();
		
		log.logln("Use method:\tpredictUsingFeatureNames (uses ArrayList<Float>)");
		float[] results = gnb.getScoresFromPrediction(new ArrayList<Float>(Arrays.asList(6f,130f)));
		
		assertEquals(4.726183760794811E-6, results[0], 0);
		assertEquals(1.4375489263329655E-4, results[1], 0);
		
		results = null;
		featuresTest.set(0,0,6.0);
		featuresTest.set(0,1,130.0);
		
		log.logln("\nUse method:\tpredictClassResults (uses NDArray)");
		results = gnb.predictClassResults(featuresTest);
		assertEquals(4.726183760794811E-6, results[0], 0);
		assertEquals(1.4375489263329655E-4, results[1], 0);

		log.logln("\nUse method:\tpredict (uses ArrayList<Float>)");
		String sResult = gnb.predict(new ArrayList<Float>(Arrays.asList(6f,130f)));
		assertEquals("Female", sResult);
		System.out.println("Result: " + sResult);
		System.out.println();
		System.out.println("*** TEST COMPLETE ***");

	}	

	@Test
	public void checkMeanVar(){
		log.logln_withClassName(G.lF,"");
		
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
		
		double[] labeled = new double[] {1,1,1,1,1,1};

		GaussianNB gnb =  new GaussianNB(GaussianNB.EMPTY_SAMPLES_ALLOW);

		//		gnb.fitSample(featuresTrain, classLabelsTrain, labelList);
		try {
		gnb.fit(featuresTrain, labeled);
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
		log.logln_withClassName(G.lF,"");

		NDArray featuresTrain = NDArray.newArray(6,2);
		NDArray featuresTest = NDArray.newArray(2,2);
		double[] labeled;

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
		
		labeled = new double[] {1,1,1,2,2,2};
//		List<String> labelList = new ArrayList<String>(Arrays.asList("Ht(ft)", "Wt(lbs)"));
	
		GaussianNB gnb =  new GaussianNB(
				GaussianNB.EMPTY_SAMPLES_ALLOW);

		//		gnb.fitSample(featuresTrain, classLabelsTrain, labelList);
		try {
		gnb.fit(featuresTrain, labeled);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printAttributeValuesAndClasses();
		gnb.printMeanVar();
		featuresTest.set(0,0,-0.8);
		featuresTest.set(0,1,-1);
		featuresTest.set(1,0,1);
		featuresTest.set(1,1,2);

		double[] result = gnb.predict(featuresTest);
		assertEquals(1.0, result[0], 0.0);
		assertEquals(2.0, result[1], 0.0);
		
		double accuracy = gnb.getAccuracyOfPredictedResults(new double[] {1.0}, result);
		assertEquals(1.0, accuracy, 0.0);
		System.out.println("Samples: " + featuresTest);
		System.out.println("Results: ");
		for (double dVal : result) {
			System.out.println (dVal);
		}
		System.out.println("Accuracy: " + accuracy);
		System.out.println();
		System.out.println("*** TEST COMPLETE ***");
	}
}
