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
import xilodyne.util.G;
import xilodyne.util.Logger;
import mikera.arrayz.INDArray;
import mikera.arrayz.NDArray;

/**
 * Gaussian Naive Bayes implementation as described in Wikipedia.org.
 * <p>
 * This implementation converts all doubles to floats to save space.
 * <p>
 * Currently only has variance calculation based on a sample of population data
 * and not a complete population. 
 * 
 * @see <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes">https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes</a>
 * @see <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html">http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html</a>
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.2 -- 4/27/2017 
 * 	changed labels/classes to features/labels;
 *  internal float storage / public double values
 * @version 0.1 -- 9/18/2016, initial implementation
 * 
 */

public class GaussianNB {

	private Logger log = new Logger();
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
	private Hashtable<Integer, TreeMap<Float, int[]>> features = new Hashtable<Integer, TreeMap<Float, int[]>>();
	private List<String> featureNames = null;  //optional, show names in output
	private int numberOfFeatures = 0;


	//classification of label, i.e. if labels are "male / female", classification would be "gender"
	private String labelClassCategory = "LABEL";

	/** mean computation of all labelValues for Label */
	private ArrayList<float[]> featuresMean = new ArrayList<float[]>();

	/** variance computation of all labelValues for Label */
	private ArrayList<double[]> featuresVariance = new ArrayList<double[]>();


	/**
	 * Instantiates a new Gaussian Naive Bayes.
	 *
	 * @param allowEmptyValues TRUE allows empty values (i.e. zero) to be added into data set
	 */
	public GaussianNB(boolean allowEmptyValues) {
		log.logln_withClassName(G.lF,"");
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
	public GaussianNB(boolean allowEmptyValues, List<String> featureNames, List<String> labelNames) {
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
		log.logln_withClassName(G.lF, "UPDATING label LIST with List<String>");
		
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
		log.logln_withClassName(G.lF, "UPDATING feature LIST with List<String>");

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
		this.moreTrainingData = true;	
		this.addNewLabelToList(trainingLabel);
		
		log.logln(G.lI, featureIndex + ", " + trainingData_OneValue + ", " + trainingLabel);
		this.updateFeatures(featureIndex, (float) trainingData_OneValue, (float)trainingLabel);

		this.totalFitEntries++;
		log.logln(G.lD, "total entries: " + this.totalFitEntries);
	}


	/**
	 * Add training data for feature set for one label
	 * 
	 * @param trainingData_SetOfValues List of float training data for one sample
	 * (List must be in same order as other feature values)
	 * @param label  associated to this class
	 */
	public void fit(List<Float> trainingData_SetOfValues, float trainingLabel) {
		this.moreTrainingData = true;
		this.addNewLabelToList(trainingLabel);
		
		log.logln(G.lI, "List size: " + trainingData_SetOfValues.size() + ", " + trainingLabel);

		for (int index = 0; index < trainingData_SetOfValues.size(); index++) {
			log.logln(index + ":" + trainingData_SetOfValues.get(index));
			this.updateFeatures(index, trainingData_SetOfValues.get(index), trainingLabel);
			log.logln(G.lD, "total entries: " + this.totalFitEntries);
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
		log.logln(G.lI, "List size: " + trainingData_SetOfValues.size() + ", " + trainingLabel);
		
		this.moreTrainingData = true;
		this.addNewLabelToList(trainingLabel);

		for (int index = 0; index < trainingData_SetOfValues.size(); index++) {
			log.logln(index + ":" + trainingData_SetOfValues.get(index));
			double val = trainingData_SetOfValues.get(index);
			this.updateFeatures(index, (float)val, (float)trainingLabel);
		}

		this.totalFitEntries++;
		log.logln(G.lD, "total entries: " + this.totalFitEntries);
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
		this.moreTrainingData = true;
		this.updateLabels(trainingLabels);
		log.logln(G.lF, "Labels: " + ArrayUtils.printArray(this.labels));

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

		log.logln(G.lF, "Fitting data...");
		log.logln(G.lI, "# of labels: " + trainingLabels.length + ", # of features: " + numberOfFeatures);
		log.log(G.lD, "INDEX\t");

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
		log.logln(G.lD, "total entries: " + this.totalFitEntries);

	}	

	/**
	 * Calculate the mean and variance. This must be done prior to any predict
	 * functions.
	 */
	private void calMeanVar() {
		log.logln(G.lF, "Calculate Mean & Var...");

		// don't calculate unless new data has been fitted
		if (this.moreTrainingData) {
			this.moreTrainingData = false;
			this.initMeanVar();

			// calculate for each label, get labels for each feature
			for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
				float[] tempMean = new float[this.features.size()];
				double[] tempVar = new double[this.features.size()];

				for (int featureKey : this.features.keySet()) {
					tempMean[featureKey] = this.calculateMean(featureKey, labelIndex);
					tempVar[featureKey] = this.calculateVarianceSample(featureKey, labelIndex, tempMean[featureKey]);

					log.log(G.lI, "Calculate mean/var for Label " + this.labels[labelIndex] + ", Feature " + featureKey);
					log.log_noTimestamp("\tMean: " + tempMean[featureKey]);
					log.logln_noTimestamp("\tVariance: " + tempVar[featureKey]);
				}

				this.featuresMean.set(labelIndex, tempMean);
				this.featuresVariance.set(labelIndex, tempVar);
			}
		}
	}
	
	/**
	 * Initializes the mean and variance variables.
	 */
	private void initMeanVar() {
		float[] tempFloat = new float[this.features.size()];
		double[] tempDouble = new double[this.features.size()];

		for (int index = 0; index < this.features.size(); index++) {
			tempFloat[index] = 0f;
			tempDouble[index] = 0;
		}

		for (int index = 0; index < this.labels.length; index++) {
			this.featuresMean.add(tempFloat);
			this.featuresVariance.add(tempDouble);
		}
	}

	
	/**
	 * Prints the mean and variance.
	 */
	public void printMeanVar() {
		this.calMeanVar();

		System.out.println();
		System.out.println("MEAN and VARIANCE (by Labels)");
		System.out.print("" + "\t\t");
		for (int featureKey = 0; featureKey < this.features.size(); featureKey++) {
			System.out.print("mean\t");
			System.out.print("var\t");
		}
		System.out.println();

		System.out.print("\tFeat:\t");
		for (int featureKey = 0; featureKey < this.features.size(); featureKey++) {
			
			//use names if available
			if ((this.featureNames == null) || (!(this.featureNames.size() == this.features.size()))) {
			System.out.print(featureKey + "\t");
			System.out.print(featureKey + "\t");
			} else {
				System.out.print(this.featureNames.get(featureKey) + "\t");
				System.out.print(this.featureNames.get(featureKey) + "\t");
			}
		}

		System.out.println();

		for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
			System.out.print(this.getClassListDisplayName() + "\t");
			
			//use names if available
			if ((this.labelNames == null) || (!(this.labelNames.size() == this.labels.length))) {
				System.out.print(this.labels[labelIndex] + "\t");
			} else {
				System.out.print(this.labelNames.get(labelIndex) + "\t");				
			}
			
			float[] tempMean = this.featuresMean.get(labelIndex);
			double[] tempVar = this.featuresVariance.get(labelIndex);
			for (int listLabelIndex = 0; listLabelIndex < this.features.size(); listLabelIndex++) {
				System.out.print(String.format("%.3f", tempMean[listLabelIndex]) + "\t");
				System.out.print(String.format("%.3f", tempVar[listLabelIndex]) + "\t");
			}
			System.out.println();
		}
		System.out.println();
	}

	/**
	 * Gets the Mean given feature and label.
	 *
	 * @param featureIndex the class index
	 * @param labelIndex the list label index
	 * @return the Mean
	 */
	public float getMean(int featureIndex, int labelIndex) {
		float[] tempMean = this.featuresMean.get(featureIndex);
		return tempMean[labelIndex];
	}

	/**
	 * Gets the variance given feature and label.
	 *
	 * @param featureIndex the feature index
	 * @param lableIndex the lable index
	 * @return the Variance
	 */
	public double getVar(int featureIndex, int lableIndex) {
		double[] tempVar = this.featuresVariance.get(featureIndex);
		return tempVar[lableIndex];
	}
	
	/**
	 * Calculate Mean for given feature.
	 *
	 * @param featureKey the feature key
	 * @param labelIndex the label index
	 * @return the Mean
	 */
	private float calculateMean(int featureKey, int labelIndex) {
		float mean = 0;
		float sumValue = 0;
		int classCount = 0;

		SortedMap<Float, int[]> tempMap = this.features.get(featureKey);
		for (Map.Entry<Float, int[]> entry : tempMap.entrySet()) {
			int[] labelCount = entry.getValue();
			int count = labelCount[labelIndex];
			sumValue += (entry.getKey() * count);
			classCount += count;
		}
		mean = sumValue / classCount;

		return mean;
	}

	/**
	 * Calculate Variance sample given feature.
	 * 
	 * http://www.wikihow.com/Calculate-Variance
	 * variance(s^2) = (for all X ( X - mean) ^2) / (n - 1) n = count of Xs
	 *
	 * @param featureKey the feature key
	 * @param labelIndex the class name index
	 * @param mean the mean
	 * @return the Variance
	 */
	private double calculateVarianceSample(int featureKey, int labelIndex, float mean) {
		double temp = 0;
		float value = 0f;
		int classCount = 0;
		SortedMap<Float, int[]> tempMap = this.features.get(featureKey);

		for (Map.Entry<Float, int[]> entry : tempMap.entrySet()) {
			int[] labelCount = entry.getValue();
			int count = labelCount[labelIndex];
			value = entry.getKey();

			// if multiple entries in same key, calculate each one
			if (count > 0) {
				for (int index = 0; index < count; index++) {
					temp += ((value - mean) * (value - mean));
					classCount++;
				}
			}
		}
		return temp / (classCount - 1);
	}

	// 
	/*
	 * variance(s^2) = (for all X ( X - mean) ^2) / n n = count of Xs
	 */

	// todo
	// austin
	/*
	 * private double calculateVariancePopulation(int listValuesIndex, int
	 * classNameIndex, float mean) { double temp = 0; float value = 0f; int
	 * classCount = 0; SortedMap<Float, int[]> tempMap =
	 * this.attributeValues.get(listValuesIndex);
	 * 
	 * for (Map.Entry<Float, int[]> entry : tempMap.entrySet()) { int[] tempBool
	 * = entry.getValue(); int count = tempBool[classNameIndex]; value =
	 * entry.getKey();
	 * 
	 * // if multiple entries in same key, calculate each one if (count > 0) {
	 * for (int index = 0; index < count; index++) { temp += ((value - mean) *
	 * (value - mean)); classCount++; } } }
	 * 
	 * return temp / (classCount); }
	 */
	// lookup one value , get value for each class
	
	


	/**
	 * Predict given list of sample set (each entry must 
	 * correspond to one index from the FEATURES hashtable)
	 *
	 * @param testingData the sample values
	 * @return the label
	 */
	public double predict_TestingSet(List<Float> testingData) {
		this.calMeanVar();
		float[] data = ArrayUtils.convertListToFloatArray(testingData);
		float[] results = this.getResultsFromSingleFeature(data);
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
		this.calMeanVar();
		Iterator<INDArray> getElement = testingData.iterator();
		float[] data = ArrayUtils.convertNDArrayEntryToFloatArray(getElement.next());
		float[] results = this.getResultsFromSingleFeature(data);
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
		this.calMeanVar();
		
		int predListCount = 0;
	//	int[] predictedLabels = new int[testingData.getShape(0)];
		log.logln_withClassName(G.LOG_FINE, "Prediction started...");
		log.logln(G.lF, "Data set size: " + testingData.getShape(0));
		log.logln(G.lD, "\nData set: " + testingData);

		// get first element
		Iterator<INDArray> getElement = testingData.iterator();
//		double[] predictedListByLabelValue = new double[predictedLabels.length];
		double[] predictedListByLabelValue = new double[testingData.getShape(0)];

		while (getElement.hasNext()) {
			float[] data = ArrayUtils.convertNDArrayEntryToFloatArray(getElement.next());
			float[] results = this.getResultsFromSingleFeature(data);
	//		predictedLabels[predListCount] = this.getPredictedLabelIndex(results);
			predictedListByLabelValue[predListCount] = (double)this.getPredictedLabel(results);

			predListCount++;
		}

//		for (int index = 0; index < predictedLabels.length; index++) {
//			predictedListByLabelValue[index] = (double)this.labels[predictedLabels[index]];
//		}
		log.logln(G.lF, "Prediction finished.");
		return predictedListByLabelValue;
	}




	
	/**
	 * Predict given Label, Feature and testing data.
	 *  
	 * @param labelIndex index of label to be checked
	 * @param featureIndex  index of feature to be checked
	 * @param testingData  value of feature
	 * @return return gaussian probability of sample value being of this label
	 */
	public float getProbabilty_OneFeature(int featureIndex, int labelIndex, float testingData) {
		this.calMeanVar();
		TreeMap<Float, int[]> tempMap = this.features.get(featureIndex);

		float Pc = this.getPcPerLabel(labelIndex, tempMap);
		float Pd_given_c = this.getGaussian_Pd_given_c(featureIndex, testingData, labelIndex, tempMap);
		log.logln_withClassName(G.lI, this.labels[labelIndex] + "\tPc: " + Pc + "\t* Pd_given_c: "
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
		this.calMeanVar();
		float[] data = ArrayUtils.convertListToFloatArray(testingData);
		double[] results = ArrayUtils.convertFloatToDoubleArray(this.getResultsFromSingleFeature(data));
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
		float[] data = ArrayUtils.convertNDArrayEntryToFloatArray(getElement.next());
		double[] results = ArrayUtils.convertFloatToDoubleArray(this.getResultsFromSingleFeature(data));
		return results;
	}
	
	/**
	 * Given single feature, determine probabilty
	 * scores for each label
	 *
	 * @param testingData the test data
	 * @return probabilty scores of feature checked
	 */
	private float[] getResultsFromSingleFeature(float[] testingData) {
		float Pc_given_d = 1, Pc = 0;
		float[] labelScores = new float[this.labels.length];

		log.log_noTimestamp(G.lD, "");
		
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
				TreeMap<Float, int[]> tempMap = this.features.get(testingIndex);
				local_Pd_given_c = this.getGaussian_Pd_given_c(testingIndex, testingData[testingIndex], 
						labelIndex, tempMap);

				log.log_noTimestamp(String.format("%.8f", local_Pd_given_c) + ")*(");

				Pd_given_c = Pd_given_c * local_Pd_given_c;
			}

			Pc_given_d = Pd_given_c * Pc;
			log.logln(String.format("%.3f", Pc) + "))\t=" + Pc_given_d);

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

		for (int featureKey = 0; featureKey < this.features.size(); featureKey++) {
				TreeMap<Float, int[]> tempMap = this.features.get(featureKey);
				uniqueLabelCount = uniqueLabelCount
						+ this.getLabelCountFromFeature(labelIndex, tempMap);
				totalLabelsCount = totalLabelsCount + this.getCountAllLabelsbyFeature(tempMap);
			}
		Pc = (float) uniqueLabelCount / totalLabelsCount;
		return Pc;

	}

	// P(c)
	/**
	 * Given label index, determine probabilty for one feature.
	 *
	 * @param labelIndex the label index
	 * @param tempMap the temp map
	 * @return the pc per label
	 */
	// className divided by all classes
	private float getPcPerLabel(int labelIndex, TreeMap<Float, int[]> tempMap) {
		float Pc;
		Pc = (float) getLabelCountFromFeature(labelIndex, tempMap) / this.getCountAllLabelsbyFeature(tempMap);
		return Pc;
	}

	/**
	 * Gets the gaussian pd given c.
	 *
	 * @param featureIndex the feature index
	 * @param testingData the testing data
	 * @param labelIndex the label index
	 * @param tempMap the temp map
	 * @return the gaussian pd given c
	 */
	private float getGaussian_Pd_given_c(int featureIndex, float testingData, int labelIndex,
			TreeMap<Float, int[]> tempMap) {
		float Pd_given_c = 0;

		float[] classMeans = this.featuresMean.get(labelIndex);
		double[] classVars = this.featuresVariance.get(labelIndex);

		float mean = classMeans[featureIndex];
		double variance_sigma_sqrd = classVars[featureIndex];

		double base = 1 / Math.sqrt(2 * Math.PI * variance_sigma_sqrd);
		double base_e = (-1 * ((testingData - mean) * (testingData - mean))) / (2 * variance_sigma_sqrd);
		base_e = Math.exp(base_e);
		Pd_given_c = (float) (base * base_e);

		return Pd_given_c;
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
			log.logln_withClassName(G.lF, "UPDATING Label list with: " + dLabelData);

			this.labels = new float[1];
			this.labels[0] = labelData;
		} else {
			//only add new labels
			if (this.getLabelIndex(labelData) == -1) {
				//add to list
				log.logln_withClassName(G.lF, "UPDATING Label list with: " + dLabelData);
				this.createNewLabelList(labelData);
				log.logln_withClassName(G.lF, "UPDATING all Features with new label.");
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
		Set<Integer> featureKeys = this.features.keySet();
		Iterator<Integer> keyIterator = featureKeys.iterator();
		
		while (keyIterator.hasNext()) {

			int featNameIndex = keyIterator.next();
			TreeMap<Float, int[]> tempMap = this.features.get(featNameIndex);
				
			Set<Float> mapKeys = tempMap.keySet();
			Iterator<Float> mapIterator = mapKeys.iterator();
			while (mapIterator.hasNext()) {
				float mapKey = mapIterator.next();
				int[] oldCounts = this.features.get(featNameIndex).get(mapKey);
				int[] newCounts = this.emptyLabelCountsForFeatures();
				System.arraycopy(oldCounts, 0, newCounts, 0, oldCounts.length);
				this.features.get(featNameIndex).replace(mapKey, newCounts);
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
			log.logln_withClassName(G.lF, "Value: " + featureValue + " not accepted.");
		} else {
			boolean featureIndexExists = this.features.containsKey(featureIndex);
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
				if (this.features.get(featureIndex).containsKey(featureValue)) {
					// update label count
					int[] labelCounts = this.features.get(featureIndex).get(featureValue);
					labelCounts[labelIndexFound]++;
					this.features.get(featureIndex).replace(featureValue, labelCounts);
				} else {
					// add new feature value
					int[] labelCounts = this.emptyLabelCountsForFeatures();
					labelCounts[labelIndexFound] = 1;
					this.features.get(featureIndex).put(featureValue, labelCounts);
				}
			} else {
				// no entries, create the feature and add the first map
				int[] labelCounts = this.emptyLabelCountsForFeatures();
				labelCounts[labelIndexFound] = 1;
				TreeMap<Float, int[]> tempMap = new TreeMap<Float, int[]>();
				tempMap.put(featureValue, labelCounts);
				this.features.put(featureIndex, tempMap);
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

		Set<Integer> featureKeys = this.features.keySet();
		
		//print the heading row
		Iterator<Integer> keyIterator = featureKeys.iterator();
		while (keyIterator.hasNext()) {
			keyIterator.next();
			System.out.print("Feature\t\t");
			for (int labelIndex = 0; labelIndex < this.labels.length; labelIndex++) {
				System.out.print("Label\t");
			}
			System.out.print("\t");
		}
		System.out.println();
		
		keyIterator = featureKeys.iterator();
		while (keyIterator.hasNext()) {

			int featNameIndex = keyIterator.next();
			//use name if available
			if ((this.featureNames == null) || (!(this.featureNames.size() == this.features.size()))) {
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
			//count++;
		}
		System.out.println();
		
		keyIterator = featureKeys.iterator();
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
		TreeMap<Float, int[]> tempMap = null;
		int[] tempLabelCount = null;
		for (int index = 0; index < maxFeatureValueCount; index++) {
			for (int featNames = 0; featNames < this.features.size(); featNames++) {

				tempMap = this.features.get(featNames);
				if (tempMap.size() > index) {
					float mapFloat = 0;
					Entry<Float, int[]> map = this.getMapAtIndex(index, tempMap);
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
		Enumeration<TreeMap<Float, int[]>> eLoop = this.features.elements();
		while (eLoop.hasMoreElements()) {
			TreeMap<Float, int[]> tempMap = eLoop.nextElement();
			if (max < tempMap.size()) {
				max = tempMap.size();
			}
		}	
		return max;
	}
	
	/**
	 * Given feature map, return an Entry at index.
	 *
	 * @param featureIndex the feature index
	 * @param tempMap the temp map
	 * @return the map at index
	 */
	//get an Entry of the data we need at index
	private Entry<Float, int[]> getMapAtIndex(int featureIndex, TreeMap<Float, int[]> tempMap){
		Set<Float> keys = tempMap.keySet();
		Iterator<Float> loop = keys.iterator();
		int count = 0;
		Entry<Float, int[]> map = null;
		
		while (loop.hasNext() && (count <= featureIndex)) {
			map = tempMap.ceilingEntry(loop.next());
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
	 * @param tempMap all the feature values for one feature
	 * @return the total count
	 */
	private int getCountAllLabelsbyFeature(SortedMap<Float, int[]> tempMap) {
		int totalLabels = 0;

		// count number of times all label has been incremented
		Collection<int[]> cindex = tempMap.values();
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
}
