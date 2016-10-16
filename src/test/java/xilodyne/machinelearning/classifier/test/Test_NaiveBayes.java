package xilodyne.machinelearning.classifier.test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import xilodyne.machinelearning.classifier.NaiveBayesClassifier;
import static org.junit.Assert.assertEquals;

public class Test_NaiveBayes {
	
	@Test
	public void checkProbabilityOneLabel() {
		//setup
		System.out.println();
		System.out.println("*** TEST *** Check Probability One Label");
		System.out.println("Testing:  P(Male|Drew)");
		System.out.println();
		List <String> labelList = new ArrayList<String>(Arrays.asList("Name"));
		List<String> classification = new ArrayList<String>(Arrays.asList("Male","Female"));
		NaiveBayesClassifier nb =  new NaiveBayesClassifier(classification, labelList);
		int labelIndex = labelList.indexOf("Name");
		nb.fit(labelIndex, "Drew", "Male");
		nb.fit(labelIndex, "Claudia", "Female");
		nb.fit(labelIndex, "Drew", "Female");
		nb.fit(labelIndex, "Drew", "Female");
		nb.fit(labelIndex, "Alberto", "Male");
		nb.fit(labelIndex, "Karin", "Female");
		nb.fit(labelIndex, "Nina", "Female");
		nb.fit(labelIndex, "Sergio", "Male");
		
		float result = nb.predictUsingFeatureNameSingleClass(classification.indexOf("Male"), labelIndex, "Drew");
		assertEquals(0.125, result, 0);
		
		result = nb.predictUsingFeatureNameSingleClass(classification.indexOf("Female"), labelIndex, "Drew");
		assertEquals(0.25, result, 0);
		System.out.println("*** TEST COMPLETE ***");
	}
	
	@Test
	public void checkProbabilityFourFeatures() {
		System.out.println();
		System.out.println("*** TEST *** Check Probability Four Features");
		System.out.println("Testing:  P(Male|Drew,>170cm,Blue Eyes, Long Hair)");
		System.out.println("Testing:  P(Female|Drew,>170cm,Blue Eyes, Long Hair)");
		List <String> labelList = new ArrayList<String>(Arrays.asList("Name",">170cm","Eye","Hair"));
		List<String> classification = new ArrayList<String>(Arrays.asList("Male","Female"));

		NaiveBayesClassifier nb = new NaiveBayesClassifier(classification, labelList);

		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Claudia","Yes","Brown","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Alberto","Yes","Brown","Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Karin","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Nina","Yes","Brown","Short")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Sergio","Yes","Blue","Long")), "Male");

		nb.printFeaturesAndClasses();
		
		float[] results = nb.predictUsingFeatureNames(new ArrayList<String>(Arrays.asList("Drew","Yes","Blue","Long")));
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

		List <String> labelList = new ArrayList<String>(Arrays.asList("Name",">170cm","Eye","Hair"));
		List<String> classification = new ArrayList<String>(Arrays.asList("Male","Female"));

		NaiveBayesClassifier nb =  new NaiveBayesClassifier(classification, labelList);

		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Claudia","Yes","Brown","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Alberto","Yes","Brown","Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Karin","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Nina","Yes","Brown","Short")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Sergio","Yes","Blue","Long")), "Male");

		nb.printFeaturesAndClasses();
		nb.determineProbabilities();

		float[] results = nb.predictUsingFeatureName(labelList.indexOf("Name"),"Drew");
		assertEquals(0.125, results[0], 0);
		assertEquals(0.25, results[1], 0);
		
		results = nb.predictUsingFeatureName(labelList.indexOf("Hair"),"Long");
		assertEquals(0.125, results[0], 0);
		assertEquals(0.5, results[1], 0);

		System.out.println("*** TEST COMPLETE ***");

	}

	
	public void XcheckProbabilityFeatureAllClasses() {
		System.out.println();
		System.out.println("*** TEST ***");
		System.out.println("Testing:  P(Male|Drew, >170cm, blue eyes, long hair)");
		System.out.println("Testing:  P(Female|Drew, >170cm, blue eyes, long hair)");
		List <String> labelList = new ArrayList<String>(Arrays.asList("Name",">170cm","Eye","Hair"));
		List<String> classification = new ArrayList<String>(Arrays.asList("Male","Female"));

		NaiveBayesClassifier nb = new NaiveBayesClassifier(classification, labelList);

		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Claudia","Yes","Brown","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Alberto","Yes","Brown","Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Karin","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Nina","Yes","Brown","Short")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Sergio","Yes","Blue","Long")), "Male");

		nb.printFeaturesAndClasses();
		
		

	//	nb.determineProbabilities();
	//	nb.determineProbabilityFeatureName(labelList.indexOf("Hair"),"Long");		

		float[] results = nb.predictUsingFeatureName(labelList.indexOf("Name"),"Drew");
		assertEquals(0.125, results[0], 0);
		assertEquals(0.25, results[1], 0);
		
		System.out.println("*** TEST COMPLETE ***");

	}

}
