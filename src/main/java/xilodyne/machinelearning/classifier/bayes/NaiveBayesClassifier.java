package xilodyne.machinelearning.classifier.bayes;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import xilodyne.util.ArrayUtils;
import xilodyne.util.data.NDArrayUtils;
import xilodyne.util.logger.Logger;
import mikera.arrayz.INDArray;
import mikera.arrayz.NDArray;

/**
 * All text values must be converted to float/double.
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
 *  internal float storage / public double values
 * @version 0.1 -- 9/18/2016, initial implementation
 * 
 */

public class NaiveBayesClassifier {

	private Logger log = new Logger("nb");
	private int totalFitEntries = 0;


	public static final boolean EMPTY_SAMPLES_ALLOW = true;
	public static final boolean EMPTY_SAMPLES_IGNORE = false;

	// which type of variance to calculate, only a sample
	// size of population data or entire population data
	// to be implemented
	// private static final boolean VARIANCE_SAMPLE_CALCULATION = true;
	// private static final boolean VARIANCE_POPULATION_CALCULATION = false;

	private boolean allowEmptySampleValues = true;
	

	/** TRUE if number of features have been loaded in the fit method for NDArray */
	private boolean featureSetFixed = false; // for multiple samples, only init once
	
	/* TRUE if training data entered. 
	 * If new data then mean / var calculation must be done prior to predict
	 */
	private boolean moreTrainingData = true; 

	private float[] labels = null;
	//boolean labelsLoad = false;
	private List<String> labelNames = null;  //optional, show names in output
	

	//Hashtable:  featureID, (TreeMap (featureValue(s), label list count, must match index of labels[]))
	private Hashtable<Integer, TreeMap<Float, int[]>> featuresList = new Hashtable<Integer, TreeMap<Float, int[]>>();
	private List<String> featureNames = null;  //optional, show names in output
	private int numberOfFeatures = 0;


	//classification of label, i.e. if labels are "male / female", classification would be "gender"
	private String labelClassCategory = "LABEL";

	/**
	 * Instantiates a new Gaussian Naive Bayes.
	 *
	 * @param allowEmptyValues TRUE allows empty values (i.e. zero) to be added into data set
	 */
	public NaiveBayesClassifier(boolean allowEmptyValues) {
		log.logln_withClassName(Logger.lF,"");
		this.allowEmptySampleValues = allowEmptyValues;
	}

	/**
	 * Instantiates a new Gaussian Naive Bayes.
	 * Optional, assign names to values, useful for printing out data
	 *
	 * @param allowEmptyValues TRUE allows empty values (i.e. zero) to be added into data set
	 * @param featureNames LIST of strings, order must match FEATURES table
	 * @param labelNames LIST of strings, order must match LABELS array
	 */
	public NaiveBayesClassifier(boolean allowEmptyValues, List<String> featureNames, List<String> labelNames) {
		this.allowEmptySampleValues = allowEmptyValues;
		this.createLabelNames(labelNames);
		this.createFeatureNames(featureNames);
	}

	
	/**
	 * Optional. Creates a List label names.
	 *
	 * @param labelList the label display name list
	 */
	private void createLabelNames(List<String> labelList) {
		log.logln_withClassName(Logger.lF, "UPDATING label LIST with List<String>");
		
		this.labelNames = new ArrayList<String>();
		for (int index = 0; index < labelList.size(); index++)
			this.labelNames.add(labelList.get(index));
	}


	/**
	 * Optional. Creates a List feature names.
	 *
	 * @param featureList the feature display name list
	 */
	private void createFeatureNames(List<String> featureList) {
		log.logln_withClassName(Logger.lF, "UPDATING feature LIST with List<String>");

		this.featureNames = new ArrayList<String>();
		for (int index = 0; index < featureList.size(); index++)
			this.featureNames.add(featureList.get(index));
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
	public void fit(int featureIndex, double trainingData_OneValue, double trainingLabel) {
		this.setMoreTrainingData(true);	
		this.addNewLabelToList(trainingLabel);
		
		log.logln(Logger.lI, featureIndex + ", " + trainingData_OneValue + ", " + trainingLabel);
		this.updateFeatures(featureIndex, (float) trainingData_OneValue, (float)trainingLabel);

		this.totalFitEntries++;
		log.logln(Logger.lD, "total entries: " + this.totalFitEntries);
	}


	/**
	 * Add training data for feature set for one label
	 * 
	 * @param trainingData_SetOfValues List of float training data for one sample
	 * (List must be in same order as other feature values)
	 * @param label  associated to this class
	 */
	public void fit(List<Float> trainingData_SetOfValues, float trainingLabel) {
		this.setMoreTrainingData(true);
		this.addNewLabelToList(trainingLabel);
		
		log.logln(Logger.lI, "List size: " + trainingData_SetOfValues.size() + ", " + trainingLabel);

		for (int index = 0; index < trainingData_SetOfValues.size(); index++) {
			log.logln(index + ":" + trainingData_SetOfValues.get(index));
			this.updateFeatures(index, trainingData_SetOfValues.get(index), trainingLabel);
			log.logln(Logger.lD, "total entries: " + this.totalFitEntries);
		}
		this.totalFitEntries++;
	}


	/**
	 * Add training data for feature set for one label
	 *
	 * @param trainingData_SetOfValues List of double training data for one sample
	 * (List must be in same order as other feature values)
	 * @param trainingLabel the label data
	 */
	public void fit(List<Double> trainingData_SetOfValues, double trainingLabel) {
		log.logln(Logger.lI, "List size: " + trainingData_SetOfValues.size() + ", " + trainingLabel);
		
		this.setMoreTrainingData(true);
		this.addNewLabelToList(trainingLabel);

		for (int index = 0; index < trainingData_SetOfValues.size(); index++) {
			log.logln(index + ":" + trainingData_SetOfValues.get(index));
			double val = trainingData_SetOfValues.get(index);
			this.updateFeatures(index, (float)val, (float)trainingLabel);
		}

		this.totalFitEntries++;
		log.logln(Logger.lD, "total entries: " + this.totalFitEntries);
	}

	/**
	 * Load in data for my samples with one or more attributes per sample
	 * 
	 * @param trainingData NDArray data structred [[val1, val2, ...], [val1, val2, ...], ... ]
	 * @param trainingLabels double[] of labeled data associated to each
	 * in NDArray
	 * @throws Exception thrown when data attributes size do not match
	 */
	public void fit(NDArray trainingData, double[] trainingLabels) throws Exception {
		this.setMoreTrainingData(true);
		this.updateLabels(trainingLabels);
		log.logln(Logger.lF, "Labels: " + ArrayUtils.printArray(this.labels));

		// update feature size only once, ignore additional features added later
		if (!this.featureSetFixed) {
			this.numberOfFeatures = trainingData.getShape(1);
			this.featureSetFixed = true;
		}

		// if loading multiple samples, make sure array sizes are the same
		if (numberOfFeatures != trainingData.getShape(1)) {
			throw new Exception("Sample data array size is not consistent: " + numberOfFeatures + " vs "
					+ trainingData.getShape(1));
		}

		Iterator<INDArray> values = trainingData.iterator();
		int count = 0;

		log.logln(Logger.lF, "Fitting data...");
		log.logln(Logger.lI, "# of labels: " + trainingLabels.length + ", # of features: " + numberOfFeatures);
		log.log(Logger.lD, "INDEX\t");

		for (int index = 0; index < numberOfFeatures; index++) {
			log.log_noTimestamp("Feature: " + (index) + "\t");
		}
		log.logln_noTimestamp(this.getClassListDisplayName());

		while (values.hasNext()) {
			INDArray value = values.next();
			log.log(count + "\t\t");

			// load each feature
			for (int index = 0; index < numberOfFeatures; index++) {
				this.updateFeatures(index, (float) value.get(index), (float) trainingLabels[count]);

				log.log_noTimestamp(value.get(index) + "\t\t");
			}
			log.logln_noTimestamp(String.valueOf(trainingLabels[count]));

			this.totalFitEntries++;
			count++;
		}
		log.logln(Logger.lD, "total entries: " + this.totalFitEntries);

	}	



	/**
	 * Predict given list of sample set (each entry must 
	 * correspond to one index from the FEATURES hashtable)
	 *
	 * @param testingData the sample values
	 * @return the label
	 */
	public double predict_TestingSet(List<Float> testingData) {
		float[] data = ArrayUtils.convertListToFloatArray(testingData);
		float[] results = this.getResultsFromFeatureSetForOneLabel(data);
		return (double) this.getPredictedLabel(results);
	}
	

	/**
	 * Predict given list of sample set (each entry must 
	 * correspond to one index from the FEATURES hashtable)
	 *
	 * @param testingData the testing data
	 * @return the label
	 */
	public double predict_TestingSet(NDArray testingData) {
		// get first element
		Iterator<INDArray> getElement = testingData.iterator();
		float[] data = NDArrayUtils.convertNDArrayEntryToFloatArray(getElement.next());
		float[] results = this.getResultsFromFeatureSetForOneLabel(data);
		return (double) this.getPredictedLabel(results);
	}
	
	


	/**
	 * Predict given a list of feature sets [[val1, val2, ...], [val1, val2, ...], ...]
	 *
	 * @param testingData the testing data
	 * @return the array of all predicted labels
	 */
	// return list of classes for each element
	public double[] predict(NDArray testingData) {
		int predListCount = 0;
	//	int[] predictedLabels = new int[testingData.getShape(0)];
		log.logln_withClassName(Logger.LOG_FINE, "Prediction started...");
		log.logln(Logger.lF, "Data set size: " + testingData.getShape(0));
		log.logln(Logger.lD, "\nData set: " + testingData);

		// get first element
		Iterator<INDArray> getElement = testingData.iterator();
//		double[] predictedListByLabelValue = new double[predictedLabels.length];
		double[] predictedListByLabelValue = new double[testingData.getShape(0)];

		while (getElement.hasNext()) {
			float[] data = NDArrayUtils.convertNDArrayEntryToFloatArray(getElement.next());
			float[] results = this.getResultsFromFeatureSetForOneLabel(data);
	//		predictedLabels[predListCount] = this.getPredictedLabelIndex(results);
			predictedListByLabelValue[predListCount] = (double)this.getPredictedLabel(results);

			predListCount++;
		}

//		for (int index = 0; index < predictedLabels.length; index++) {
//			predictedListByLabelValue[index] = (double)this.labels[predictedLabels[index]];
//		}
		log.logln(Logger.lF, "Prediction finished.");
		return predictedListByLabelValue;
	}




	
	/**
	 * Predict given Label, Feature and testing data.
	 *  
	 * @param labelIndex index of label to be checked
	 * @param featuresIndex  index of feature to be checked
	 * @param testingData  value of feature
	 * @return return gaussian probability of sample value being of this label
	 */
	public float getProbabilty_OneFeature(int featuresIndex, int labelIndex, float testingData) {
		TreeMap<Float, int[]> featureValues = this.featuresList.get(featuresIndex);

		float Pc = this.getPcPerLabel(labelIndex, featureValues);
		float Pd_given_c = this.getPd_given_c(featuresIndex, testingData, labelIndex, featureValues);
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
	public double[] getProbabilityScores_TestingSet(List<Float> testingData) {
		float[] data = ArrayUtils.convertListToFloatArray(testingData);
		double[] results = ArrayUtils.convertFloatToDoubleArray(this.getResultsFromFeatureSetForOneLabel(data));
		return results;
	}
	
	/**
	 * Gets the probability scores testing set.
	 *
	 * @param testingData the testing data
	 * @return the probability scores testing set
	 */
	public double[] getProbabilityScores_TestingSet(NDArray testingData) {
		// get first element
		Iterator<INDArray> getElement = testingData.iterator();
		float[] data = NDArrayUtils.convertNDArrayEntryToFloatArray(getElement.next());
		double[] results = ArrayUtils.convertFloatToDoubleArray(this.getResultsFromFeatureSetForOneLabel(data));
		return results;
	}
	
	/**
	 * Given single feature, determine probabilty
	 * scores for each label
	 *
	 * @param testingData the test data
	 * @return probabilty scores of feature checked
	 */
	private float[] getResultsFromFeatureSetForOneLabel(float[] testingData) {
		float Pc_given_d = 1, Pc = 0;
		float[] labelScores = new float[this.labels.length];

		log.log_noTimestamp(Logger.lD, "");
		
		log.log("Predict label using values:\t");
		int index = 0;
		for (float f : testingData) {
			log.log_noTimestamp(String.valueOf(index));
			log.log_noTimestamp(":");
			log.log_noTimestamp(f + "\t");
			index++;
		}
		log.logln_noTimestamp("");
		
		for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
			// each entry equal to 1 to avoid zeroing out
			float Pd_given_c = 1;
			Pc = this.getPcForAllValuesByLabel(labelIndex);
			log.log(this.labels[labelIndex] + "\t(");

			for (int testingIndex = 0; testingIndex < testingData.length; testingIndex++) {
				log.log_noTimestamp(testingData[testingIndex] + ":");

				float local_Pd_given_c = 0;
				TreeMap<Float, int[]> featureValues = this.featuresList.get(testingIndex);
				local_Pd_given_c = this.getPd_given_c(testingIndex, testingData[testingIndex], 
						labelIndex, featureValues);

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
	private float getPredictedLabel(float[] results) {
		// find the greatest value
		float getMax = 0;
		int labelMax = 0;
		for (int index = 0; index < this.labels.length; index++) {
			if (results[index] > getMax) {
				getMax = results[index];
				labelMax = index;
			}
		}
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

		for (int featuresIndex = 0; featuresIndex < this.featuresList.size(); featuresIndex++) {
				TreeMap<Float, int[]> featureValues = this.featuresList.get(featuresIndex);
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
	// className divided by all classes
	private float getPcPerLabel(int labelIndex, TreeMap<Float, int[]> featureValues) {
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
	private float getPd_given_c(int featureIndex, float testingData, int labelIndex,
			TreeMap<Float, int[]> featureValues) {
		float Pd_given_c = 0;
		// p(d | cj )
		// given class, determine number of times featureName has className /
		// total # className
		if (featureValues.containsKey(testingData)) {
			Pd_given_c = (float) this.getLabelCountFromFeatureValues(labelIndex, testingData, featureValues)
					/ this.getLabelCountFromFeature(labelIndex, featureValues);
		} else {
			System.out.println("No instances of " + testingData + ".");
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
	
	// find how many times a feature is associated to a label
	private int getLabelCountFromFeatureValues(int labelIndex, float featureName, SortedMap<Float, int[]> featureValues) {
		int[] classCounts = featureValues.get(featureName);
		return classCounts[labelIndex];
	}

	/**
	 * Update label list with new labels.
	 *
	 * @param labelData the label data
	 */
	//and update all feature with new label counts
	private void updateLabels(double[] labelData) {
		for (double d : labelData) {
			this.addNewLabelToList(d);
		}
	}



	/**
	 * Update labels.
	 * Add only unique labels.  If adding new label,
	 * keep the same ordering in the array.
	 *
	 * @param dLabelData the d label data
	 */
	private void addNewLabelToList(double dLabelData) {
		float labelData = (float) dLabelData;
		
		if (this.labels == null) {
			log.logln_withClassName(Logger.lF, "UPDATING Label list with: " + dLabelData);

			this.labels = new float[1];
			this.labels[0] = labelData;
		} else {
			//only add new labels
			if (this.getLabelIndex(labelData) == -1) {
				//add to list
				log.logln_withClassName(Logger.lF, "UPDATING Label list with: " + dLabelData);
				this.createNewLabelList(labelData);
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
	private void createNewLabelList(float labelData) {
		float[] tempList = this.labels.clone();
		this.labels = new float[tempList.length + 1];

		System.arraycopy(tempList, 0, this.labels, 0, tempList.length);
		//add value to last entry in list, index starts at 0
		this.labels[tempList.length] = labelData;
	}
	
	/**
	 * For each feature index, for each feature value, update
	 * the label counts to reflect the number of labels.
	 */
	//if label added to list, the feature count needs to be updated
	private void addNewLabelToAllFeatures(){
		Set<Integer> featuresKeys = this.featuresList.keySet();
		Iterator<Integer> keyIterator = featuresKeys.iterator();
		
		while (keyIterator.hasNext()) {

			int featNameIndex = keyIterator.next();
			TreeMap<Float, int[]> featureValues = this.featuresList.get(featNameIndex);
				
			Set<Float> mapKeys = featureValues.keySet();
			Iterator<Float> mapIterator = mapKeys.iterator();
			while (mapIterator.hasNext()) {
				float mapKey = mapIterator.next();
				int[] oldCounts = this.featuresList.get(featNameIndex).get(mapKey);
				int[] newCounts = this.emptyLabelCountsForFeatures();
				System.arraycopy(oldCounts, 0, newCounts, 0, oldCounts.length);
				this.featuresList.get(featNameIndex).replace(mapKey, newCounts);
			}
		}
	}
	

	/**
	 * Given label value, find index.
	 *
	 * @param valueToFind the value to find
	 * @return the label index
	 */
	private int getLabelIndex(float valueToFind) {
		int labelIndex = -1;

		for (int index = 0; index < this.labels.length; index++) {
			if (valueToFind == this.labels[index]) {
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
	private void updateFeatures(int featureIndex, float featureValue, float labelToIncrement) {
		if ((featureValue == 0) && !this.allowEmptySampleValues) {
			log.logln_withClassName(Logger.lF, "Value: " + featureValue + " not accepted.");
		} else {
			boolean featureIndexExists = this.featuresList.containsKey(featureIndex);
			int labelIndexFound = this.getLabelIndex(labelToIncrement);
			// get the label index to update
			// get the feature index to update the list of feature values &
			// counts in TreeMap
			// this.updateLabels(newLabel);

			// if first time, no entry in features, create map and add
			// if featureindex found, and map found, update, replace
			// if featureindex found, and no map found and put
			// else and newfeature to amp
			if (featureIndexExists) {
				if (this.featuresList.get(featureIndex).containsKey(featureValue)) {
					// update label count
					int[] labelCounts = this.featuresList.get(featureIndex).get(featureValue);
					labelCounts[labelIndexFound]++;
					this.featuresList.get(featureIndex).replace(featureValue, labelCounts);
				} else {
					// add new feature value
					int[] labelCounts = this.emptyLabelCountsForFeatures();
					labelCounts[labelIndexFound] = 1;
					this.featuresList.get(featureIndex).put(featureValue, labelCounts);
				}
			} else {
				// no entries, create the feature and add the first map
				int[] labelCounts = this.emptyLabelCountsForFeatures();
				labelCounts[labelIndexFound] = 1;
				TreeMap<Float, int[]> featureValues = new TreeMap<Float, int[]>();
				featureValues.put(featureValue, labelCounts);
				this.featuresList.put(featureIndex, featureValues);
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

		Set<Integer> featuresKeys = this.featuresList.keySet();
		
		//print the heading row
		Iterator<Integer> keyIterator = featuresKeys.iterator();
		while (keyIterator.hasNext()) {
			keyIterator.next();
			System.out.print("Feature\t\t");
			for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
				System.out.print("Label\t");
			}
			System.out.print("\t");
		}
		System.out.println();
		
		keyIterator = featuresKeys.iterator();
		while (keyIterator.hasNext()) {

			int featNameIndex = keyIterator.next();
			//use name if available
			if ((this.featureNames == null) || (!(this.featureNames.size() == this.featuresList.size()))) {
				System.out.print(featNameIndex + "\t\t");
			} else {
				System.out.print(this.featureNames.get(featNameIndex) + "\t\t");
			}
			for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
				//use names if available
				if ((this.labelNames == null) || (!(this.labelNames.size() == this.labels.length))) {
					System.out.print(this.labels[labelIndex] + "\t");
				} else {
					System.out.print(this.labelNames.get(labelIndex) + "\t");
				}
			}
			System.out.print("\t");
		}
		System.out.println();
		
		keyIterator = featuresKeys.iterator();
		while (keyIterator.hasNext()) {
			keyIterator.next();
			System.out.print("----------\t");
			for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
				System.out.print("-----\t");
			}
			System.out.print("\t");
		}
		System.out.println();

		int maxFeatureValueCount = this.getMaxCountFeatureValues();
		TreeMap<Float, int[]> featureValues = null;
		int[] tempLabelCount = null;
		for (int index = 0; index < maxFeatureValueCount; index++) {
			for (int featNames = 0; featNames < this.featuresList.size(); featNames++) {

				featureValues = this.featuresList.get(featNames);
				if (featureValues.size() > index) {
					Entry<Float, int[]> map = this.getMapAtIndex(index, featureValues);
					System.out.print(map.getKey() + "\t\t");
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
		Enumeration<TreeMap<Float, int[]>> eLoop = this.featuresList.elements();
		while (eLoop.hasMoreElements()) {
			TreeMap<Float, int[]> featureValues = eLoop.nextElement();
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
	private Entry<Float, int[]> getMapAtIndex(int featureIndex, TreeMap<Float, int[]> featureValues){
		Set<Float> keys = featureValues.keySet();
		Iterator<Float> loop = keys.iterator();
		int count = 0;
		Entry<Float, int[]> map = null;
		
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
	private int getLabelCountFromFeature(int labelIndex, SortedMap<Float, int[]> trainData) {
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
	private int getCountAllLabelsbyFeature(SortedMap<Float, int[]> featureValues) {
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
