package xilodyne.machinelearning.classifier.bayes;

import java.util.Collection;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.Map.Entry;

import xilodyne.util.ArrayUtils;
import xilodyne.util.logger.Logger;

/**
 * Text Based.  Only text values (not numerics) are used.
 * 
 * Naive Bayes for classification implementation as described
 * by Prof Eamonn Keogh, UCR.
 * @see <a href="http://www.cs.ucr.edu/~eamonn/CE/Bayesian%20Classification%20withInsect_examples.pdf">http://www.cs.ucr.edu/~eamonn/CE/Bayesian%20Classification%20withInsect_examples.pdf</a> 
 * <p>
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.4 - 1/29/2018 - reflect xilodyne util changes
 * @version 0.2 -- 5/9/2017
 * 	changed labels/classes to features/labels;
 *  allow text input instead of numeric
 * @version 0.1 -- 9/18/2016, initial implementation
 * 
 */
public class NaiveBayesClassifier_UsingTextValues {


	private Logger log = new Logger("NBTV");
	private int totalFitEntries = 0;


	public static final boolean EMPTY_SAMPLES_ALLOW = true;
	public static final boolean EMPTY_SAMPLES_IGNORE = false;

	// which type of variance to calculate, only a sample
	// size of population data or entire population data
	// to be implemented
	// private static final boolean VARIANCE_SAMPLE_CALCULATION = true;
	// private static final boolean VARIANCE_POPULATION_CALCULATION = false;

	private boolean allowEmptySampleValues = true;
	
	
	/* TRUE if training data entered. 
	 * If new data then mean / var calculation must be done prior to predict
	 */
	private boolean moreTrainingData = true; 

	private String[] labels = null;	

	//Hashtable:  featureID, (TreeMap (featureValue(s), label list count, must match index of labels[]))
	private Hashtable<String, TreeMap<String, int[]>> featuresList = new Hashtable<String, TreeMap<String, int[]>>();

	//classification of label, i.e. if labels are "male / female", classification would be "gender"
	private String labelClassCategory = "LABEL";

	/**
	 * Instantiates a new Gaussian Naive Bayes.
	 *
	 * @param allowEmptyValues TRUE allows empty values (i.e. zero) to be added into data set
	 */
	public NaiveBayesClassifier_UsingTextValues(boolean allowEmptyValues) {
		log.logln_withClassName(Logger.lF,"");
		this.allowEmptySampleValues = allowEmptyValues;
	}


	

	/**
	 * Sets the label class category.
	 *
	 * @param newName the new label class category (i.e. if labels are "male / female"
	 * then class category would be "gender")
	 */
	public void setLabelClassCategory(String newName) {
		this.labelClassCategory = newName;
	}


	/**
	 * Returns the class list display name.
	 *
	 * @return the class list display name
	 */
	public String getClassListDisplayName() {
		return this.labelClassCategory;
	}


	/**
	 * Add training data one feature at a time.
	 *
	 * @param featureIndex the feature index, where to place the data in the features list
	 * @param trainingData_OneValue the feature value
	 * @param trainingLabel the label data
	 */
	public void fit(String feature, String trainingData_OneValue, String trainingLabel) {
		this.setMoreTrainingData(true);	
		this.addNewLabelToList(trainingLabel);
		
		log.logln(Logger.lI, feature + ", " + trainingData_OneValue + ", " + trainingLabel);
		this.updateFeatures(feature, trainingData_OneValue, trainingLabel);

		this.totalFitEntries++;
		log.logln(Logger.lD, "total entries: " + this.totalFitEntries);
	}



	/**
	 * Predict given list of sample set (each entry must 
	 * correspond to one index from the FEATURES hashtable)
	 *
	 * @param testingData the sample values
	 * @return the label
	 */
	public String predict_TestingSet(Hashtable<String, String> testingData) {
		float[] results = this.getResultsFromFeatureSetForOneLabel(testingData);
		return this.getPredictedLabel(results);
	}
	

	

	
	/**
	 * Predict given Label, Feature and testing data.
	 *  
	 * @param labelName index of label to be checked
	 * @param featureIndex  index of feature to be checked
	 * @param testingData  value of feature
	 * @return return gaussian probability of sample value being of this label
	 */
	public float getProbabilty_OneFeature(String featureName, String labelName, String testingData) {
		TreeMap<String, int[]> featureValues = this.featuresList.get(featureName.toUpperCase());
		int labelIndex = this.getLabelIndex(labelName);
		float Pc = this.getPcPerLabel(labelIndex, featureValues);
		float Pd_given_c = this.getPd_given_c(labelIndex, testingData, featureValues);
		log.logln_withClassName(Logger.lI, this.labels[labelIndex] + "\tPc: " + Pc + "\t* Pd_given_c: "
				+ Pd_given_c + "\t= " + Pd_given_c * Pc);

		return Pd_given_c * Pc;
	}

	/**
	 * Gets the probability scores testing set.
	 *
	 * @param testingData the testing data
	 * @return the probability scores testing set
	 */
	//return the calculations for each label
	public double[] getProbabilityScores_TestingSet(Hashtable<String, String> testingData) {
		float[] results = this.getResultsFromFeatureSetForOneLabel(testingData); 
		return ArrayUtils.convertFloatToDoubleArray(results);
	}
	
	
	/**
	 * Given single feature, determine probabilty
	 * scores for each label
	 *
	 * @param testingData the test data
	 * @return probabilty scores of feature checked
	 */
	private float[] getResultsFromFeatureSetForOneLabel(Hashtable<String, String> testingData) {
		float Pc_given_d = 1, Pc = 0;
		float[] labelScores = new float[this.labels.length];

		log.log_noTimestamp(Logger.lD, "");
		
		log.log("Predict label using values:\t");
		Collection<String> valueSet = testingData.values();
		Iterator<String> values = valueSet.iterator();
 		while (values.hasNext()) {
			log.log_noTimestamp(values.next() + "\t");
		}
		log.logln_noTimestamp("");
		
		for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
			// each entry equal to 1 to avoid zeroing out
			float Pd_given_c = 1;
			Pc = this.getPcForAllValuesByLabel(labelIndex);
			log.log(this.labels[labelIndex] + "\t(");

			Collection<String> featuresKeys = testingData.keySet();
			Iterator<String> featureNames = featuresKeys.iterator();
			while (featureNames.hasNext()) {
				String featureName = featureNames.next();
				String featureValue = testingData.get(featureName);
				log.log_noTimestamp(featureName + "/" +featureValue +":");

				float local_Pd_given_c = 0;
				TreeMap<String, int[]> featureValues = this.featuresList.get(featureName.toUpperCase());
				local_Pd_given_c = this.getPd_given_c(labelIndex, featureValue.toLowerCase(), featureValues);

				log.log_noTimestamp(String.format("%.8f", local_Pd_given_c) + ")*(");

				Pd_given_c = Pd_given_c * local_Pd_given_c;
			}

			Pc_given_d = Pd_given_c * Pc;
			log.logln_noTimestamp(String.format("%.3f", Pc) + "))\t=" + Pc_given_d);

			labelScores[labelIndex] = Pc_given_d;
		}
		return labelScores;
	}


	/**
	 * Gets the accuracy of predicted results.
	 *
	 * @param testingLabels the test data
	 * @param predictedLabels the results data
	 * @return the accuracy of predicted results
	 */
	public double getAccuracyOfPredictedResults(double[] testingLabels, double[] predictedLabels) {
		int count = 0;
		for (int index = 0; index < testingLabels.length; index++) {
			if (testingLabels[index] == predictedLabels[index])
				count++;
		}

		return (double) count / testingLabels.length;
	}

	/**
	 * Given list of label counts for a feature value,
	 * find label with greatest count.
	 *
	 * @param results the results
	 * @return the predicted label
	 */
	private String getPredictedLabel(float[] results) {
		// find the greatest value
		float getMax = 0;
		int labelMax = 0;
		for (int index = 0; index < this.labels.length; index++) {
			if (results[index] > getMax) {
				getMax = results[index];
				labelMax = index;
			}
		}
		log.logln(Logger.lD,  "Label Predicted: " + this.labels[labelMax]);
		return this.labels[labelMax];
	}
	

	/**
	 * Given label index, determine probability for all features.
	 *
	 * @param labelIndex the label index
	 * @return the pc for all values by label
	 */
	private float getPcForAllValuesByLabel(int labelIndex) {
		float Pc = 0;
		int uniqueLabelCount = 0;
		int totalLabelsCount = 0;

		//for (int featureKey = 0; featureKey < this.features.size(); featureKey++) {
		Set<String> keySet = this.featuresList.keySet();
		for (String featuresKey: keySet) {
			TreeMap<String, int[]> featureValues = this.featuresList.get(featuresKey);
				uniqueLabelCount = uniqueLabelCount
						+ this.getLabelCountFromFeature(labelIndex, featureValues);
				totalLabelsCount = totalLabelsCount + this.getCountAllLabelsbyFeature(featureValues);
			}
		Pc = (float) uniqueLabelCount / totalLabelsCount;
		return Pc;

	}

	// P(c)
	/**
	 * Given label index, determine probabilty for one feature.
	 *
	 * @param labelIndex the label index
	 * @param featureValues the temp map
	 * @return the pc per label
	 */
	// className divided bgetPcPerLabely all classes
	private float getPcPerLabel(int labelIndex, TreeMap<String, int[]> featureValues) {
		float Pc;
		Pc = (float) getLabelCountFromFeature(labelIndex, featureValues) / this.getCountAllLabelsbyFeature(featureValues);
		return Pc;
	}

	/*
	 * p(cj | d) = p(d | cj ) p(cj) ---------------- p(d)
	 */

	/**
	 * Gets the pd given c.
	 *
	 * @param className
	 *            the class name
	 * @param featureValue
	 *            the feature name
	 * @param featureValues
	 *            the temp map
	 * @return the pd given c
	 */
	private float getPd_given_c(int labelIndex, String featureValue, SortedMap<String, int[]> featureValues) {
		float Pd_given_c = 0;
		// p(d | cj )
		// given class, determine number of times featureName has className /
		// total # className
		if (featureValues.containsKey(featureValue.toLowerCase())) {
			Pd_given_c = (float) this.getLabelCountFromFeatureValues(labelIndex, featureValue.toLowerCase(), featureValues)
					/ this.getLabelCountFromFeature(labelIndex, featureValues);
		} else {
			System.out.println("No instances of " + featureValue.toLowerCase() + ".");
		//	throw new Exception("Not in dictionary: " + featureName);
		/*	try {
			//	throw new Exception("No instances of " + featureName + ".");
				throw new Exception();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			*/
		}

		// System.out.println("(" + className + "|" + featureName + ") is " +
		// ((float) Pd_given_c * getPc(className)));

		return Pd_given_c;
	}
	
	/**
	 * Gets the feature freq by class.
	 *
	 * @param className
	 *            the class name
	 * @param featureName
	 *            the feature name
	 * @param featureValues
	 *            the temp map
	 * @return the feature freq by class
	 */
	// find how many times a feature is associated to a label
	private int getLabelCountFromFeatureValues(int labelIndex, String featureName, SortedMap<String, int[]> featureValues) {
		int[] classCounts = featureValues.get(featureName);
		return classCounts[labelIndex];
	}

	/**
	 * Gets the class frequency from features.
	 *
	 * @param className
	 *            the class name
	 * @param featureValues
	 *            the temp map
	 * @return the class frequency from features
	 */
	// get total number of times className has been incremented
	@SuppressWarnings("unused")
	private int getClassFrequencyFromFeatures(String className, SortedMap<String, int[]> featureValues) {
		int classFrequency = 0;
		Collection<int[]> cLoop = featureValues.values();
		Iterator<int[]> iLoop = cLoop.iterator();
		while (iLoop.hasNext()) {
			int[] iValues = iLoop.next();
			classFrequency = classFrequency + iValues[this.getLabelIndex(className)];
		}
		return classFrequency;
	}


	


	/**
	 * Update labels.
	 * Add only unique labels.  If adding new label,
	 * keep the same ordering in the array.
	 *
	 * @param dLabelData the d label data
	 */
	private void addNewLabelToList(String label) {
		
		if (this.labels == null) {
			log.logln_withClassName(Logger.lF, "UPDATING Label list with: " + label);

			this.labels = new String[1];
			this.labels[0] = label.toUpperCase();
		} else {
			//only add new labels
			if (this.getLabelIndex(label) == -1) {
				//add to list
				log.logln_withClassName(Logger.lF, "UPDATING Label list with: " + label);
				this.createNewLabelList(label);
				log.logln_withClassName(Logger.lF, "UPDATING all Features with new label.");
				this.addNewLabelToAllFeatures();
			}
		}
	}
	
	/**
	 * Increment label list.
	 *
	 * @param labelData the label data
	 */
	//add new entry to label list but keep same order
	private void createNewLabelList(String label) {
		String[] tempList = this.labels.clone();
		this.labels = new String[tempList.length + 1];

		System.arraycopy(tempList, 0, this.labels, 0, tempList.length);
		//add value to last entry in list, index starts at 0
		this.labels[tempList.length] = label.toUpperCase();
	}
	
	/**
	 * For each feature index, for each feature value, update
	 * the label counts to reflect the number of labels.
	 */
	//if label added to list, the feature count needs to be updated
	private void addNewLabelToAllFeatures(){
		Set<String> featuresKeys = this.featuresList.keySet();
		Iterator<String> keyIterator = featuresKeys.iterator();
		
		while (keyIterator.hasNext()) {

			String feature = keyIterator.next();
			TreeMap<String, int[]> featureValues = this.featuresList.get(feature);
				
			Set<String> mapKeys = featureValues.keySet();
			Iterator<String> mapIterator = mapKeys.iterator();
			while (mapIterator.hasNext()) {
				String mapKey = mapIterator.next();
				int[] oldCounts = this.featuresList.get(feature).get(mapKey);
				int[] newCounts = this.emptyLabelCountsForFeatures();
				System.arraycopy(oldCounts, 0, newCounts, 0, oldCounts.length);
				this.featuresList.get(feature).replace(mapKey, newCounts);
			}
		}
	}
	

	/**
	 * Given label value, find index.
	 *
	 * @param valueToFind the value to find
	 * @return the label index
	 */
	private int getLabelIndex(String valueToFind) {
		int labelIndex = -1;

		for (int index = 0; index < this.labels.length; index++) {
			if (valueToFind.equalsIgnoreCase(this.labels[index])) {
				labelIndex = index;
				break;
			}
		}
		return labelIndex;
	}

	/**
	 * Update features by incrementing the appropriate label count
	 * associated to the feature value of the feature index.
	 *
	 * @param featureIndex the feature index
	 * @param featureValue the feature value
	 * @param labelToIncrement the label to increment
	 */
	// increment the appropriate label[index] for the feature
	private void updateFeatures(String feature, String featureValue, String labelToIncrement) {
		if (featureValue.isEmpty() && !this.allowEmptySampleValues) {
			log.logln_withClassName(Logger.lF, "Value: " + featureValue + " not accepted.");
		} else {
			boolean featureExists = this.featuresList.containsKey(feature.toUpperCase());
			int labelIndexFound = this.getLabelIndex(labelToIncrement);
			// get the label index to update
			// get the feature index to update the list of feature values &
			// counts in TreeMap
			// this.updateLabels(newLabel);

			// if first time, no entry in features, create map and add
			// if feature found, and map found, update, replace
			// if feature found, and no map found and put
			// else and newfeature to amp
			if (featureExists) {
				if (this.featuresList.get(feature.toUpperCase()).containsKey(featureValue.toLowerCase())) {
					// update label count
					int[] labelCounts = this.featuresList.get(feature.toUpperCase()).get(featureValue.toLowerCase());
					labelCounts[labelIndexFound]++;
					this.featuresList.get(feature.toUpperCase()).replace(featureValue.toLowerCase(), labelCounts);
				} else {
					// add new feature value
					int[] labelCounts = this.emptyLabelCountsForFeatures();
					labelCounts[labelIndexFound] = 1;
					this.featuresList.get(feature.toUpperCase()).put(featureValue.toLowerCase(), labelCounts);
				}
			} else {
				// no entries, create the feature and add the first map
				int[] labelCounts = this.emptyLabelCountsForFeatures();
				labelCounts[labelIndexFound] = 1;
				TreeMap<String, int[]> featureValues = new TreeMap<String, int[]>();
				featureValues.put(featureValue.toLowerCase(), labelCounts);
				this.featuresList.put(feature.toUpperCase(), featureValues);
			}
		}
	}

	/**
	 * Create an int array full of zeros.
	 *
	 * @return the int[]
	 */
	private int[] emptyLabelCountsForFeatures(){
		int[] labelCounts = new int[this.labels.length];
		for (int loop = 0; loop < labelCounts.length; loop++)
			labelCounts[loop] = 0;
		
		return labelCounts;
	}
	

	/**
	 * Prints the features and labels.
	 */
	public void printFeaturesAndLabels() {
		//if featureNames and labelsNames not loaded, this print float values
		System.out.println();
		System.out.println("Label counts by feature.\n");

		Set<String> featuresKeys = this.featuresList.keySet();
		
		//print the heading row
		Iterator<String> feature = featuresKeys.iterator();
		while (feature.hasNext()) {
			feature.next();
			System.out.print("Feature\t\t");
			for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
				System.out.print("Label\t");
			}
			System.out.print("\t");
		}
		System.out.println();
		
		feature = featuresKeys.iterator();
		while (feature.hasNext()) {

			//String feat = feature.next();
			//use name if available
				System.out.print(feature.next() + "\t\t");
			for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
				//use names if available
					System.out.print(this.labels[labelIndex] + "\t");
			}
			System.out.print("\t");
			//count++;
		}
		System.out.println();
		
		feature = featuresKeys.iterator();
		while (feature.hasNext()) {
			feature.next();
			System.out.print("----------\t");
			for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
				System.out.print("-----\t");
			}
			System.out.print("\t");
		}
		System.out.println();

		int maxFeatureValueCount = this.getMaxCountFeatureValues();
		TreeMap<String, int[]> featureValues = null;
		int[] tempLabelCount = null;
		for (int index = 0; index < maxFeatureValueCount; index++) {
		//	for (int featNames = 0; featNames < this.features.size(); featNames++) {
			feature = featuresKeys.iterator();	
			while (feature.hasNext()) {

			//	System.out.println("feature search: " + )
				featureValues = this.featuresList.get(feature.next());
				if (featureValues.size() > index) {
					//Set<String> maps = tempMap.keySet();
					//Iterator<String> mapKeys = maps.iterator();
				
					Entry<String, int[]> map = this.getMapAtIndex(index, featureValues);
					//Entry<String, int[]> map = 
					// this.getMapAtIndex(index, tempMap, tempLabelCount,
					// mapFloat );
					System.out.print(map.getKey() + "\t\t");
				//	System.out.print(mapFloat + "\t");
					tempLabelCount = map.getValue();
					for (int countIndex = 0; countIndex < tempLabelCount.length; countIndex++) {
						System.out.print(tempLabelCount[countIndex] + "\t");					
					}
					System.out.print("\t");
				} else {
					System.out.print("-\t\t");
					for (int countIndex = 0; countIndex < this.labels.length; countIndex++) {
						System.out.print("-\t");
					}
					System.out.print("\t");
				}
				
			}
			System.out.println();
		}
	}
	
	/**
	 * Determine which feature map has the most values
	 *
	 * @return the max count feature values
	 */
	//determine the which feature has the most values associated to it
	private int getMaxCountFeatureValues(){
		int max = 0;
		Enumeration<TreeMap<String, int[]>> eLoop = this.featuresList.elements();
		while (eLoop.hasMoreElements()) {
			TreeMap<String, int[]> featureValues = eLoop.nextElement();
			if (max < featureValues.size()) {
				max = featureValues.size();
			}
		}	
		return max;
	}
	
	/**
	 * Given feature map, return an Entry at index.
	 *
	 * @param featureIndex the feature index
	 * @param featureValues the temp map
	 * @return the map at index
	 */
	//get an Entry of the data we need at index
	private Entry<String, int[]> getMapAtIndex(int featureIndex, TreeMap<String, int[]> featureValues){
		Set<String> keys = featureValues.keySet();
		Iterator<String> loop = keys.iterator();
		int count = 0;
		Entry<String, int[]> map = null;
		
		while (loop.hasNext() && (count <= featureIndex)) {
			map = featureValues.ceilingEntry(loop.next());
			count++;
		}
		return map;
	}


	/**
	 * Gets total number of times label has been incremented for a feature
	 *
	 * @param labelIndex the label index
	 * @param trainData the train data
	 * @return the label frequency 
	 */
	private int getLabelCountFromFeature(int labelIndex, SortedMap<String, int[]> trainData) {
		int labelFrequency = 0;
		Collection<int[]> cIndex = trainData.values();
		Iterator<int[]> iIndex = cIndex.iterator();
		while (iIndex.hasNext()) {
			int[] iValues = iIndex.next();
			labelFrequency = labelFrequency + iValues[labelIndex];
		}
		return labelFrequency;
	}

	/**
	 * Sum total label counts for a feature.
	 *
	 * @param featureValues all the feature values for one feature
	 * @return the total count
	 */
	private int getCountAllLabelsbyFeature(TreeMap<String, int[]> featureValues) {
		int totalLabels = 0;

		// count number of times all label has been incremented
		Collection<int[]> cindex = featureValues.values();
		Iterator<int[]> iindex = cindex.iterator();
		while (iindex.hasNext()) {
			int[] iValues = iindex.next();
			for (int index = 0; index < iValues.length; index++) {
				totalLabels = totalLabels + iValues[index];
			}
		}

		return totalLabels;
	}

	/**
	 * Gets the given featureIndex and treemap, return the corresponding
	 * feature value
	 *
	 * @param featureMap the feature map
	 * @param index the index
	 * @return the feature key at index
	 */
	@SuppressWarnings("unused")
	private float getFeatureValueAtIndex(TreeMap<Float, int[]> featureMap, int index) {
		float key = 0;

		int indexCount = 0;
		for (Map.Entry<Float, int[]> entry : featureMap.entrySet()) {
			if (indexCount == index) {
				key = entry.getKey();
			}
			indexCount++;
		}
		return key;
	}



	
	public int getFitCount() {
		return this.totalFitEntries;
	}




	public boolean isMoreTrainingData() {
		return moreTrainingData;
	}




	public void setMoreTrainingData(boolean moreTrainingData) {
		this.moreTrainingData = moreTrainingData;
	}
}
