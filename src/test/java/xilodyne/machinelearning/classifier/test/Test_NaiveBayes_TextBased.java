package xilodyne.machinelearning.classifier.test;


import java.util.Hashtable;

import org.junit.Test;

import xilodyne.machinelearning.classifier.bayes.NaiveBayesClassifier_UsingTextValues;
import xilodyne.util.G;
//import xilodyne.util.Logger;
import static org.junit.Assert.assertEquals;

public class Test_NaiveBayes_TextBased {
	
	//private Logger log = new Logger();

	public Test_NaiveBayes_TextBased(){
		// G.setLoggerLevel(G.LOG_OFF);
		//G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		G.setLoggerLevel(G.LOG_DEBUG);

	}
	
	@Test
	public void checkProbabilityOneLabel() {
		//setup
		System.out.println();
		System.out.println("*** TEST *** Check Probability One Label");
		System.out.println("Testing:  P(Male|Drew)");
		System.out.println("Testing:  P(Female|Drew)");
		System.out.println();
		
		NaiveBayesClassifier_UsingTextValues nb =  new NaiveBayesClassifier_UsingTextValues(
				NaiveBayesClassifier_UsingTextValues.EMPTY_SAMPLES_IGNORE);
		nb.fit("Name", "Drew", "Male");
		nb.fit("Name", "Claudia", "Female");
		nb.fit("Name", "Drew", "Female");
		nb.fit("Name", "Drew", "Female");
		nb.fit("Name", "Alberto", "Male");
		nb.fit("Name", "Karin", "Female");
		nb.fit("Name", "Nina", "Female");
		nb.fit("Name", "Sergio", "Male");
		
		nb.printFeaturesAndLabels();
		
		double result = 0;
		result = nb.getProbabilty_OneFeature("Name", "Male", "Drew");
		assertEquals(0.125, result, 0);
		
		result = nb.getProbabilty_OneFeature("Name", "Female", "Drew");
		assertEquals(0.25, result, 0);
		System.out.println("*** TEST COMPLETE ***");
	}
	
	@Test
	public void checkProbabilityFourFeatures() {
		System.out.println();
		System.out.println("*** TEST *** Check Probability Four Features");
		System.out.println("Testing:  P(Male|Drew,>170cm,Blue Eyes, Long Hair)");
		System.out.println("Testing:  P(Female|Drew,>170cm,Blue Eyes, Long Hair)");
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
		
		nb.fit(featureCategories[0], "Karin", labels[1]);
		nb.fit(featureCategories[1], "No", labels[1]);
		nb.fit(featureCategories[2], "Blue", labels[1]);
		nb.fit(featureCategories[3], "Long", labels[1]);
		
		nb.fit(featureCategories[0], "Nina", labels[1]);
		nb.fit(featureCategories[1], "Yes", labels[1]);
		nb.fit(featureCategories[2], "Brown", labels[1]);
		nb.fit(featureCategories[3], "Short", labels[1]);
		
		nb.fit(featureCategories[0], "Sergio", labels[0]);
		nb.fit(featureCategories[1], "Yes", labels[0]);
		nb.fit(featureCategories[2], "Blue", labels[0]);
		nb.fit(featureCategories[3], "Long", labels[0]);


/*		("Drew","No","Blue","Short"), "Male"
		("Claudia","Yes","Brown","Long"), "Female"
		("Drew","No","Blue","Long"), "Female"
		("Drew","No","Blue","Long"), "Female"
		("Alberto","Yes","Brown","Short"), "Male"
		("Karin","No","Blue","Long"), "Female"
		("Nina","Yes","Brown","Short"), "Female"
		("Sergio","Yes","Blue","Long"), "Male"
*/
		nb.printFeaturesAndLabels();
		String predictedLabel = null;
		Hashtable<String, String> testingData_OneSet = new Hashtable<String, String>();
		testingData_OneSet.put(featureCategories[0], "Drew");
		testingData_OneSet.put(featureCategories[1], "Yes");
		testingData_OneSet.put(featureCategories[2], "Blue");
		testingData_OneSet.put(featureCategories[3], "Long");
		predictedLabel = nb.predict_TestingSet(testingData_OneSet);
		System.out.println("Predicted: " + predictedLabel);
		
		//assertEquals("FEMALE", predictedLabel, 0);
		
		double[] results = nb.getProbabilityScores_TestingSet(testingData_OneSet);
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
		
		nb.fit(featureCategories[0], "Karin", labels[1]);
		nb.fit(featureCategories[1], "No", labels[1]);
		nb.fit(featureCategories[2], "Blue", labels[1]);
		nb.fit(featureCategories[3], "Long", labels[1]);
		
		nb.fit(featureCategories[0], "Nina", labels[1]);
		nb.fit(featureCategories[1], "Yes", labels[1]);
		nb.fit(featureCategories[2], "Brown", labels[1]);
		nb.fit(featureCategories[3], "Short", labels[1]);
		
		nb.fit(featureCategories[0], "Sergio", labels[0]);
		nb.fit(featureCategories[1], "Yes", labels[0]);
		nb.fit(featureCategories[2], "Blue", labels[0]);
		nb.fit(featureCategories[3], "Long", labels[0]);
		
		nb.printFeaturesAndLabels();


		double result = 0;
		result = nb.getProbabilty_OneFeature("Name", "Male", "Drew");
		assertEquals(0.125, result, 0);
		
		result = nb.getProbabilty_OneFeature("Name", "Female", "Drew");
		assertEquals(0.25, result, 0);
		
		result = nb.getProbabilty_OneFeature("Hair", "Male", "Long");
		assertEquals(0.125, result, 0);
		
		result = nb.getProbabilty_OneFeature("Hair", "Female", "Long");
		assertEquals(0.5, result, 0);

		System.out.println("*** TEST COMPLETE ***");

	}


}
