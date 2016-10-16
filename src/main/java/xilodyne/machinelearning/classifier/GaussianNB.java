package xilodyne.machinelearning.classifier;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import xilodyne.util.G;
import xilodyne.util.Logger;

import mikera.arrayz.INDArray;
import mikera.arrayz.NDArray;

/**
 * Gaussian Naive Bayes implementation as described in Wikipedia.org.
 * <p>
 * This implementation uses two types of data instantiation:  text and/or floats.
 * <p>
 * Class methods are aligned with sklearn.naive_bayes.GaussianNB implementation. 
 * In particular it is possible, using the vectorz NDArray, to use "fit" and "predict"
 * methods to load data and check samples.
 * <p>
 * Currently only has variance calculation based on a sample of population data
 * and not a complete population. 
 * 
 * @see <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes">https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes</a>
 * @see <a href="http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html">http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html</a>
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 * 
 */

public class GaussianNB {

	private Logger log = new Logger();

	public static final boolean EMPTY_SAMPLES_ALLOW = true;
	public static final boolean EMPTY_SAMPLES_IGNORE = false;

	// which type of variance to calculate, only a sample
	// size of population data or entire population data
	// to be implemented
	// private static final boolean VARIANCE_SAMPLE_CALCULATION = true;
	// private static final boolean VARIANCE_POPULATION_CALCULATION = false;

	private boolean allowEmptySampleValues = true;
	private boolean initPerformed = false; // for multiple samples, only init once

	private int dimSize = 0;
	/** If loading in the attributes as List, the class must be in a
	 * separate data structure
	 * 
	 * If loading in NDArray and labeledData, class will be determined from labeled data
	 * 
	 */
	/**
	 * Label list: list of different types of classes: name, gender, height, etc
	 * 
	 * Attributes: for each label, attributes associated to label
	 * for label GENDER: male, female 
	 * 
	 * Classification: which label set to be classified
	 * determine class (ie male or female) given feature sets (name, height,
	 * etc.) features, classes <names, int[] (male, female), <male, female>
	 */

	// features = Name, count of name for each class (male, female)
	// classes = id, class name
	/** The class list. */
	// private SortedMap<String, int[]> features = null; //hold features for
	// each label
	private List<String> classList = null; // eg male, female; what we want to
											// identify
	private String classListDisplayName = "CLASS";

	/** The attribute list. */
	private List<String> attributeNames = null; // different sets of labels:
												// height, shoe size, ...

	/** The attribute values. */
	private ArrayList<SortedMap<Float, int[]>> attributeValues = null; 
	// for each label, a sorted list of each values,
	// boolean[] = if class has this value

	// private ArrayList<SortedMap<String, int[]>> labelFeatures = null; 
	//for
	// each label, a sorted list of each features, int[]= # of classes for
	// counting

	// array list index = className Index, float[] = size of labelList
	/** mean computation of all labelValues for Class */
	private ArrayList<float[]> classMeanValuePerAttribute = null; 

	/** variance computation of all labelValues for Class */
	private ArrayList<double[]> classVarianceValuePerAttribute = null; 

	public GaussianNB(boolean allowEmptyValues) {
		this.init(allowEmptyValues);
	}

	public GaussianNB(boolean allowEmptyValues, List<String> newClassList, List<String> newLabelList) {
		this.init(allowEmptyValues);
		this.initClassList(newClassList);
		this.initAttributeNames(newLabelList);
		this.initMeanVar();
	}

	private void init(boolean allowEmptyValues) {

		this.allowEmptySampleValues = allowEmptyValues;

		this.classList = new ArrayList<String>();
		this.attributeNames = new ArrayList<String>();
		this.attributeValues = new ArrayList<SortedMap<Float, int[]>>();
		this.classMeanValuePerAttribute = new ArrayList<float[]>();
		this.classVarianceValuePerAttribute = new ArrayList<double[]>();
	}

	private void initClassList(List<String> newClassList) {
		log.logln_withClassName(G.lF, "UPDATING CLASS LIST with List<String>");

		for (int loop = 0; loop < newClassList.size(); loop++)
			this.updateClasses(newClassList.get(loop));
	}

	private void initClassList(double[] classLabels) {
		// get unique labels and create load class Labels
		log.logln_withClassName(G.lF, "UPDATING CLASS LIST with double[]");

		for (int loop = 0; loop < classLabels.length; loop++)
			this.updateClasses(String.valueOf(classLabels[loop]));
	}

	private void initAttributeNames(List<String> newAttributeNames) {
		log.logln(G.lF, "");
		log.logln("UPDATING ATTRIBUTE NAMES");

		for (int loop = 0; loop < newAttributeNames.size(); loop++) {
			this.updateAttributeNames(newAttributeNames.get(loop));
			this.attributeValues.add((SortedMap<Float, int[]>) new TreeMap<Float, int[]>());
		}
	}

	private void initMeanVar() {
		// based upon lastList size, init the Mean and Variance
		// for each class, init float[] for size of labelList
		float[] tempFloat = new float[this.attributeNames.size()];
		double[] tempDouble = new double[this.attributeNames.size()];

		for (int loop = 0; loop < this.attributeNames.size(); loop++) {
			tempFloat[loop] = 0f;
			tempDouble[loop] = 0;
		}

		for (int classNameIndex = 0; classNameIndex < this.classList.size(); classNameIndex++) {
			this.classMeanValuePerAttribute.add(tempFloat);
			this.classVarianceValuePerAttribute.add(tempDouble);
		}
	}

	private void initLabelList(NDArray newValues) {
		this.initPerformed = true;
		log.logln_noTimestamp(G.lF, "");
		log.logln("UPDATING LABEL LIST");
		log.logln(G.lI, "Class size: " + this.classList.size());
		log.logln("ND Array element dimension: " + newValues.getShape(1));

		// initially no value for labelList, use index number
		for (int loop = 0; loop < newValues.getShape(1); loop++) {
			this.updateAttributeNames(String.valueOf(loop));
			this.attributeValues.add((SortedMap<Float, int[]>) new TreeMap<Float, int[]>());
		}

		log.logln("Value size: " + this.attributeValues.size() + ",\tLabel size: " + this.attributeNames.size());
	}

	public void setClassListDisplayName(String newName) {
		this.classListDisplayName = newName;
	}

	public void setClassNames(String[] newNames) {
		for (int index = 0; index < this.classList.size(); index++) {
			this.classList.set(index, newNames[index]);
		}
	}

	public void setLabelNames(String[] newNames) {
		for (int index = 0; index < this.attributeNames.size(); index++) {
			this.attributeNames.set(index, newNames[index]);
		}
	}

	public String getClassListDisplayName() {
		return this.classListDisplayName;
	}

	// if a new attribute, add to list
	/**
	 * Update Attribute Names.
	 *
	 * @param attributeName
	 *            the label name
	 */
	private void updateAttributeNames(String attributeName) {
		if (!attributeNames.contains(attributeName)) {
			attributeNames.add(attributeName);
			log.logln(G.lI, "AttributeNames[" + attributeNames.indexOf(attributeName) + "] " + attributeName);
		}
	}

	/**
	 * Update classes.
	 *
	 * @param className
	 *            the class name
	 */
	// if a new class, add to list
	private void updateClasses(String className) {
		if (!classList.contains(className)) {
			classList.add(className);
			log.logln(G.lI, "ClassList[" + classList.indexOf(className) + "] " + className);
		}
	}


	/**
	 * Add the sample data with the associated label class
	 * @param labelValueIndex attribute to be updated
	 * @param labelValue  value of attribute
	 * @param className  name of class
	 * 		the class name (i.e. label data) for this sample
	 */
	public void fit(int labelValueIndex, float labelValue, String className) {
		log.logln(G.lI, labelValueIndex + ", " + labelValue + ", " + className);
		this.updateLabelValues(labelValueIndex, labelValue, this.getIndexOfClassName(className));
		this.calMeanVar();
	}


	/**
	 * Load in data for single sample with multiple attributes
	 * 
	 * @param newValues list of attribute values for one sample
	 * @param className  associated to this class
	 */
	public void fit(List<Float> newValues, String className) {
		log.logln(G.lI, "List size: " + newValues.size() + ", " + className);

		for (int loop = 0; loop < newValues.size(); loop++) {
			log.logln(loop + ":" + newValues.get(loop));
			this.updateLabelValues(loop, newValues.get(loop), this.getIndexOfClassName(className));
		}
		this.calMeanVar();
	}

	/**
	 * Load in data for my samples with one or more attributes per sample
	 * 
	 * @param data NDArray containing array of [val1, val2, ...]
	 * @param classLabels double[] of labeled data associated to each
	 * in NDArray
	 * @throws Exception thrown when data attributes size do not match
	 */
	public void fit(NDArray data, double[] classLabels) throws Exception {

		if (!this.initPerformed) {
			this.initClassList(classLabels);
			this.initLabelList(data);
			this.initMeanVar();

			this.dimSize = data.getShape(1);
		}

		// if loading multiple samples, make sure array sizes are the same
		if (dimSize != data.getShape(1)) {
			throw new Exception("Sample data array size is not consistent: " + dimSize + " vs " + data.getShape(1));
		}

		Iterator<INDArray> values = data.iterator();
		int count = 0;

		log.logln(G.lF, "Fitting data...");
		log.logln(G.lI, "List size: " + dimSize + ", # of label classes: " + classLabels.length);

		log.log(G.lD, "INDEX\t");

		for (int index = 0; index < dimSize; index++) {
			log.log_noTimestamp("label " + (index + 1) + "\t");
		}
		log.log_noTimestamp(this.getClassListDisplayName());

		while (values.hasNext()) {
			INDArray value = values.next();
			log.log_noTimestamp(count + "\t");

			int classIndex = this.getIndexOfClassName(String.valueOf(classLabels[count]));
			for (int index = 0; index < dimSize; index++) {
				this.updateLabelValues(index, (float) value.get(index), classIndex);

				log.log_noTimestamp(value.get(index) + "\t");

			}

			log.log_noTimestamp(this.classList.get(classIndex));

			count++;
		}
		// update the values
		this.calMeanVar();
	}

	// for each class, get each label and calculate Mean and Variance
	private void calMeanVar() {
		log.logln(G.lF, "Calculate Mean & Var...");

		for (int classIndex = 0; classIndex < this.classList.size(); classIndex++) {
			float[] tempMean = new float[this.attributeNames.size()];
			double[] tempVar = new double[this.attributeNames.size()];

			for (int attributeNameIndex = 0; attributeNameIndex < this.attributeNames.size(); attributeNameIndex++) {
				tempMean[attributeNameIndex] = this.calculateMean(attributeNameIndex, classIndex);
				tempVar[attributeNameIndex] = this.calculateVarianceSample(attributeNameIndex, classIndex,
						tempMean[attributeNameIndex]);

				log.log(G.lI, "Calculate mean/var for Class " + this.classList.get(classIndex) + " Label "
						+ this.attributeNames.get(attributeNameIndex));
				log.log_noTimestamp("\tMean: " + tempMean[attributeNameIndex]);
				log.logln_noTimestamp("\tVariance: " + tempVar[attributeNameIndex]);
			}

			this.classMeanValuePerAttribute.set(classIndex, tempMean);
			this.classVarianceValuePerAttribute.set(classIndex, tempVar);
		}
	}

	public void printMeanVar() {

		System.out.println();
		System.out.println("MEAN and VARIANCE (Class by Attribute Name)");
		System.out.print("" + "\t\t");
		for (int loop = 0; loop < this.attributeNames.size(); loop++) {
			System.out.print("mean\t");
			System.out.print("var\t");
		}
		System.out.println();

		System.out.print("\tAttr\t");
		for (int loop = 0; loop < this.attributeNames.size(); loop++) {
			System.out.print(this.attributeNames.get(loop) + "\t");
			System.out.print(this.attributeNames.get(loop) + "\t");
		}

		System.out.println();

		for (int classIndex = 0; classIndex < this.classList.size(); classIndex++) {
			System.out.print(this.getClassListDisplayName() + "\t");
			System.out.print(this.classList.get(classIndex) + "\t");
			float[] tempMean = this.classMeanValuePerAttribute.get(classIndex);
			double[] tempVar = this.classVarianceValuePerAttribute.get(classIndex);
			for (int listLabelIndex = 0; listLabelIndex < this.attributeNames.size(); listLabelIndex++) {
				System.out.print(String.format("%.3f", tempMean[listLabelIndex]) + "\t");
				System.out.print(String.format("%.3f", tempVar[listLabelIndex]) + "\t");
			}
			System.out.println();
		}
		System.out.println();
	}

	public float getMean(int classIndex, int listLabelIndex) {
		float[] tempMean = this.classMeanValuePerAttribute.get(classIndex);
		return tempMean[listLabelIndex];
	}

	public double getVar(int classIndex, int listLabelIndex) {
		double[] tempVar = this.classVarianceValuePerAttribute.get(classIndex);
		return tempVar[listLabelIndex];
	}

	private float calculateMean(int listValuesIndex, int classNameIndex) {
		float mean = 0;
		float sumValue = 0;
		int classCount = 0;
		// if (LOGLEVEL == Level.ALL)
		SortedMap<Float, int[]> tempMap = this.attributeValues.get(listValuesIndex);
		for (Map.Entry<Float, int[]> entry : tempMap.entrySet()) {
			int[] tempBool = entry.getValue();
			int count = tempBool[classNameIndex];
			sumValue += (entry.getKey() * count);
			classCount += count;
		}
		mean = sumValue / classCount;

		return mean;
	}

	/*
	 * variance(s^2) = (for all X ( X - mean) ^2) / (n - 1) n = count of Xs
	 */
	private double calculateVarianceSample(int listValuesIndex, int classNameIndex, float mean) {
		double temp = 0;
		float value = 0f;
		int classCount = 0;
		SortedMap<Float, int[]> tempMap = this.attributeValues.get(listValuesIndex);

		for (Map.Entry<Float, int[]> entry : tempMap.entrySet()) {
			int[] tempBool = entry.getValue();
			int count = tempBool[classNameIndex];
			value = entry.getKey();

			// if multiple entries in same key, calculate each one
			if (count > 0) {
				for (int loop = 0; loop < count; loop++) {
					temp += ((value - mean) * (value - mean));
					classCount++;
				}
			}
		}

		return temp / (classCount - 1);
	}

	// http://www.wikihow.com/Calculate-Variance
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
	 * for (int loop = 0; loop < count; loop++) { temp += ((value - mean) *
	 * (value - mean)); classCount++; } } }
	 * 
	 * return temp / (classCount); }
	 */
	// lookup one value , get value for each class
	/**
	 * Predict given Class, Label and sample Value
	 *  
	 * @param classNameIndex index of class to be checked
	 * @param labelIndex  index of attribute to be checked
	 * @param sampleValue  value of attribute
	 * @return return gaussian probability of sample value being of this class
	 */
	public float predictSingleLabelSingleClass(int classNameIndex, int labelIndex, float sampleValue) {
		SortedMap<Float, int[]> tempMap = this.attributeValues.get(labelIndex);

		float Pc = this.getPcPerLabel(classNameIndex, tempMap);
		float Pd_given_c = this.getGaussian_Pd_given_c(classNameIndex, labelIndex, sampleValue, tempMap);
		log.logln_withClassName(G.lI, this.classList.get(classNameIndex) + "\tPc: " + Pc + "\t* Pd_given_c: "
				+ Pd_given_c + "\t= " + Pd_given_c * Pc);

		return Pd_given_c * Pc;
	}

	public String predict(List<Float> sampleValues) {
		// return most likely class
		float[] results = this.predictUsingFeatureNames(sampleValues);
		int classIndex = this.getPredictedClass(results);
		log.logln_withClassName(G.lI, "Most likely: " + this.classList.get(classIndex));
		return this.classList.get(classIndex);
	}

	// return list of classes for each element
	public double[] predict(NDArray sampleValues) {
		int predListCount = 0;
		int[] predictedListByClassNameIndex = new int[sampleValues.getShape(0)];
		log.logln_withClassName(G.LOG_FINE, "Prediction started...");
		log.logln(G.lF, "Data set size: " + sampleValues.getShape(0));
		log.logln(G.lD, "\nData set: " + sampleValues);

		// get first element
		Iterator<INDArray> getElement = sampleValues.iterator();
		while (getElement.hasNext()) {
			List<Float> alValues = new ArrayList<Float>();

			INDArray value = getElement.next();
			for (int index = 0; index < sampleValues.getShape(1); index++) {
				alValues.add((float) value.get(index));
			}

			float[] results = this.predictUsingFeatureNames(alValues);
			predictedListByClassNameIndex[predListCount] = this.getPredictedClass(results);
			predListCount++;
		}

		double[] predictedListByClassName = new double[predictedListByClassNameIndex.length];
		for (int i = 0; i < predictedListByClassName.length; i++) {
			predictedListByClassName[i] = Double.valueOf(this.classList.get(predictedListByClassNameIndex[i]));
		}
		log.logln(G.lF, "Prediction finished.");
		return predictedListByClassName;
	}

	// return list of classes for each element
	public float[] predictClassResults(NDArray sampleValues) {
		List<Float> alValues = new ArrayList<Float>();

		// get first element
		Iterator<INDArray> getElement = sampleValues.iterator();
		INDArray value = getElement.next();
		for (int index = 0; index < sampleValues.getShape(1); index++) {
			alValues.add((float) value.get(index));
		}

		return this.predictUsingFeatureNames(alValues);
	}

	public float[] getScoresFromPrediction(List<Float> sampleValues) {
		return this.predictUsingFeatureNames(sampleValues);
	}
	
	// featureLabel order
	private float[] predictUsingFeatureNames(List<Float> sampleValues) {
		float Pc_given_d = 1, Pc = 0;
		float[] classScores = new float[this.classList.size()];

		log.log_noTimestamp(G.lD, "");
		
		log.log("Predict Classes using sample values:\t");
		int index = 0;
		for (float f : sampleValues) {
			log.log_noTimestamp(this.attributeNames.get(index));
			log.log_noTimestamp(":");
			log.log_noTimestamp(f + "\t");
			index++;
		}
		log.logln_noTimestamp("");

		for (int classListIndex = 0; classListIndex < this.classList.size(); classListIndex++) {
			// each entry equal to 1 to avoid zeroing out
			float Pd_given_c = 1;
			Pc = this.getPcForAllValues(classListIndex, sampleValues);
			log.log(this.classList.get(classListIndex) + "\t(");

			for (int sampleValuesIndex = 0; sampleValuesIndex < sampleValues.size(); sampleValuesIndex++) {
				log.log_noTimestamp(sampleValues.get(sampleValuesIndex) + ":");

				float local_Pd_given_c = 0;
				SortedMap<Float, int[]> tempMap = this.attributeValues.get(sampleValuesIndex);
				local_Pd_given_c = this.getGaussian_Pd_given_c(classListIndex, sampleValuesIndex,
						sampleValues.get(sampleValuesIndex), tempMap);

				log.log_noTimestamp(String.format("%.8f", local_Pd_given_c) + ")*(");

				Pd_given_c = Pd_given_c * local_Pd_given_c;
			}

			Pc_given_d = Pd_given_c * Pc;
			log.logln(String.format("%.3f", Pc) + "))\t=" + Pc_given_d);

			classScores[classListIndex] = Pc_given_d;
		}
		return classScores;
	}

	public double getAccuracyOfPredictedResults(double[] testData, double[] resultsData) {
		int count = 0;
		for (int index = 0; index < testData.length; index++) {
			if (testData[index] == resultsData[index])
				count++;
		}

		return (double) count / testData.length;
	}

	private int getPredictedClass(float[] results) {
		// find the greatest value
		float getMax = 0;
		int classMax = 0;
		for (int index = 0; index < this.classList.size(); index++) {
			if (results[index] > getMax) {
				getMax = results[index];
				classMax = index;
			}
		}
		return classMax;
	}

	private float getPcForAllValues(int classNameIndex, List<Float> sampleValues) {
		float Pc = 0;
		int classPerFeatureCount = 0;
		int classCountAll = 0;

		for (int listLabelIndex = 0; listLabelIndex < this.attributeValues.size(); listLabelIndex++) {
			for (int sampleValueIndex = 0; sampleValueIndex < sampleValues.size(); sampleValueIndex++) {
				SortedMap<Float, int[]> tempMap = this.attributeValues.get(sampleValueIndex);
				classPerFeatureCount = classPerFeatureCount
						+ this.getClassFrequencyFromFeatures(classNameIndex, tempMap);
				classCountAll = classCountAll + this.getClassCountLabelFeature(tempMap);
			}
		}
		Pc = (float) classPerFeatureCount / classCountAll;
		return Pc;

	}

	// P(c)
	// className divided by all classes
	private float getPcPerLabel(int classNameIndex, SortedMap<Float, int[]> tempMap) {
		float Pc;
		Pc = (float) getClassFrequencyFromFeatures(classNameIndex, tempMap) / this.getClassCountLabelFeature(tempMap);
		return Pc;
	}

	public float getGaussian_Pd_given_c(int classNameIndex, int labelListIndex, float sampleValue,
			SortedMap<Float, int[]> tempMap) {
		float Pd_given_c = 0;

		// float mean = 0;
		// float variance = 0;

		float[] classMeans = this.classMeanValuePerAttribute.get(classNameIndex);
		double[] classVars = this.classVarianceValuePerAttribute.get(classNameIndex);

		float mean = classMeans[labelListIndex];
		double variance_sigma_sqrd = classVars[labelListIndex];

		// double base = 1 /Math.sqrt( 2 * Math.PI *
		// (Math.sqrt(variance_sigma_sqrd)));
		double base = 1 / Math.sqrt(2 * Math.PI * variance_sigma_sqrd);
		double base_e = (-1 * ((sampleValue - mean) * (sampleValue - mean))) / (2 * variance_sigma_sqrd);
		base_e = Math.exp(base_e);
		Pd_given_c = (float) (base * base_e);
		// System.out.println("mean: " + mean + ",\tvar: " + variance_sigma_sqrd
		// +",\tbase: "+ base + "\texp: " + base_e);
		// System.out.println("mean: " + mean + ",\tvar: " + variance_sigma_sqrd
		// +",\tPd_given_c: "+Pd_given_c);

		return Pd_given_c;
	}

	/*
	 * // assuming that newFeatures size = featureLabel size AND newFeature
	 * matches // featureLabel order public void fit(List<String> newFeatures,
	 * String className) { for (int loop = 0; loop < newFeatures.size(); loop++)
	 * { this.updateFeatures(loop, newFeatures.get(loop),
	 * this.getIndexOfClassName(className)); } }
	 * 
	 * // load in string array of words, associated to one class public void
	 * fit(int featureLabelIndex, String[] observedFeatures, String className) {
	 * int iIndex = this.getIndexOfClassName(className); for (String s :
	 * observedFeatures) { this.updateFeatures(featureLabelIndex, s, iIndex); }
	 * 
	 * }
	 */

	// add feature to the correct label, if exists, increment the classNameIndex
	private void updateLabelValues(int labelValueIndex, float labelValue, int classNameIndex) {

		if ((labelValue == 0) && !this.allowEmptySampleValues) {
			log.logln_withClassName(G.lF, "Value: " + labelValue + " not accepted.");

		} else {

			if (this.attributeValues.get(labelValueIndex).containsKey(labelValue)) {
				int[] featureValues = this.attributeValues.get(labelValueIndex).get(labelValue);

				featureValues[classNameIndex] = featureValues[classNameIndex] + 1;
				this.attributeValues.get(labelValueIndex).replace(labelValue, featureValues);
				/*
				 * if (LOGLEVEL == Level.FINE || LOGLEVEL == Level.INFO ||
				 * LOGLEVEL == Level.ALL) { for (int loop = 0; loop <
				 * featureValues.length; loop++) { System.out.println(labelValue
				 * + "[" + loop + "]:\t" + featureValues[loop]); } }
				 */

			} else {
				int[] classCountEmpty = new int[classList.size()];
				for (int loop = 0; loop < classList.size(); loop++)
					classCountEmpty[loop] = 0;

				classCountEmpty[classNameIndex] = 1;
				this.attributeValues.get(labelValueIndex).put(labelValue, classCountEmpty);

				/*
				 * if (LOGLEVEL == Level.FINE || LOGLEVEL == Level.INFO ||
				 * LOGLEVEL == Level.ALL) { for (int loop = 0; loop <
				 * classCountEmpty.length; loop++) {
				 * System.out.println(labelValue + "[" + loop + "]:\t" +
				 * classCountEmpty[loop]); } }
				 */
			}
		}
	}

	/*
	 * // add feature to the correct label, if exists, increment the
	 * classNameIndex private void updateLabelValues(int labelValueIndex, float
	 * labelValue, int classNameIndex) { if
	 * (this.labelValues.get(labelValueIndex).containsKey(labelValue)) {
	 * boolean[] labelValuesClasses =
	 * this.labelValues.get(labelValueIndex).get(labelValue);
	 * 
	 * 
	 * labelValuesClasses[classNameIndex] = true;
	 * this.labelValues.get(labelValueIndex).replace(labelValue,
	 * labelValuesClasses); if (LOGLEVEL == Level.ALL) { for (int loop = 0; loop
	 * < labelValuesClasses.length; loop++) { System.out.println( labelValue
	 * +"["+ loop + "]:\t" + labelValuesClasses[loop]); } }
	 * 
	 * 
	 * 
	 * } else { boolean[] classEmpty = new boolean[classList.size()]; for (int
	 * loop = 0; loop < classList.size(); loop++) classEmpty[loop] = false;
	 * 
	 * classEmpty[classNameIndex] = true;
	 * this.labelValues.get(labelValueIndex).put(labelValue, classEmpty); if
	 * (LOGLEVEL == Level.ALL) { for (int loop = 0; loop < classEmpty.length;
	 * loop++) { System.out.println( labelValue +"["+ loop + "]:\t" +
	 * classEmpty[loop]); } } }
	 * 
	 * }
	 */
	/**
	 * Prints the features and classes.
	 */
	public void printAttributeValuesAndClasses() {
		System.out.println();
		System.out.println("DATA VALUES COUNT (BY CLASS)");
		for (int nameIndex = 0; nameIndex < this.attributeNames.size(); nameIndex++) {
			System.out.print("Attr\t");
			for (int classIndex = 0; classIndex < this.classList.size(); classIndex++) {
				System.out.print(this.getClassListDisplayName() + "\t");
			}
			System.out.print("\t");
		}
		System.out.println();

		for (int nameIndex = 0; nameIndex < this.attributeNames.size(); nameIndex++) {
			// first line
			System.out.print(this.attributeNames.get(nameIndex) + "\t");

			for (int classIndex = 0; classIndex < this.classList.size(); classIndex++) {
				System.out.print(this.classList.get(classIndex) + "\t");
			}
			System.out.print("\t");
		}
		System.out.println();

		for (int nameIndex = 0; nameIndex < this.attributeNames.size(); nameIndex++) {
			System.out.print("-----\t");
			for (int classIndex = 0; classIndex < this.classList.size(); classIndex++) {
				System.out.print("-----\t");
			}
			System.out.print("\t");
		}
		System.out.println();

		for (int namesIndex = 0; namesIndex < this.getLargestListFeatureSize(); namesIndex++) {
			for (int attValueIndex = 0; attValueIndex < this.attributeValues.size(); attValueIndex++) {
				// for each attribute value, show the featureName
				float labelVal = this.getFeatureKeyAtIndex(this.attributeValues.get(attValueIndex), namesIndex);
				if (labelVal == 0) {
					System.out.print("---\t");
				} else {
					// System.out.print(String.format("%.3f", labelVal) + "\t");
					System.out.print(labelVal + "\t");
				}

				int[] values = this.getFeatureValueAtIndex(this.attributeValues.get(attValueIndex), namesIndex);
				for (int classCount = 0; classCount < values.length; classCount++) {
					if (labelVal == 0) {
						System.out.print("\t");
					} else {
						System.out.print(values[classCount] + "\t");
					}
				}
				System.out.print("\t");
			}
			System.out.println();
		}
	}

	private int getIndexOfClassName(String className) {
		if (!classList.contains(className))
			System.out.println("*** ERROR: " + className + " not in list. ***");
		return classList.indexOf(className);
	}

	private int getLargestListFeatureSize() {
		int largeIndex = 0;
		for (int loop = 0; loop < this.attributeValues.size(); loop++) {
			if (largeIndex < this.attributeValues.get(loop).size()) {
				largeIndex = this.attributeValues.get(loop).size();
			}
		}
		return largeIndex;
	}

	/*
	 * 
	 * 
	 * public void determineProbabilities() { if (LOGLEVEL == Level.INFO) {
	 * 
	 * System.out.println(); System.out.println("Probabilty for each feature");
	 * 
	 * for (int labelIndex = 0; labelIndex < this.labelList.size();
	 * labelIndex++) { // first line
	 * System.out.print(this.labelList.get(labelIndex) + "\t");
	 * 
	 * for (int classIndex = 0; classIndex < this.classList.size();
	 * classIndex++) { System.out.print(this.classList.get(classIndex) + "\t");
	 * } System.out.print("\t"); } System.out.println(); }
	 * 
	 * for (int featuresIndex = 0; featuresIndex < this
	 * .getLargestListFeatureSize(); featuresIndex++) { for (int
	 * labelFeatureIndex = 0; labelFeatureIndex < this.labelFeatures .size();
	 * labelFeatureIndex++) { // for each label, show the featureName String
	 * featureName =
	 * this.getFeatureKeyAtIndex(this.labelFeatures.get(labelFeatureIndex),
	 * featuresIndex); if (LOGLEVEL == Level.INFO) System.out.print(featureName
	 * + "\t");
	 * 
	 * SortedMap<String, int[]> tempMap =
	 * this.labelFeatures.get(labelFeatureIndex); for (int classNameIndex = 0;
	 * classNameIndex < this.classList.size(); classNameIndex++) { float
	 * Pc_given_d=0, Pd_given_c=0, Pc = 0; //
	 * System.out.println(loopFeatures+":"+tempMap.size()); if
	 * (tempMap.isEmpty() || ((tempMap.size()-1) < featuresIndex)) { if
	 * (LOGLEVEL == Level.INFO) System.out.print("\t"); } else { Pd_given_c =
	 * this.getPd_given_c(classList.get(classNameIndex), featureName, tempMap);
	 * Pc = this.getPcPerLabel(classList.get(classNameIndex), tempMap);
	 * Pc_given_d = Pd_given_c * Pc;
	 * 
	 * if (LOGLEVEL == Level.INFO) System.out.print(String.format("%.3f",
	 * Pc_given_d) + "\t"); } } //this.getProbabilities(featureLabelIndex,
	 * featureName) if (LOGLEVEL == Level.INFO) System.out.print("\t"); } if
	 * (LOGLEVEL == Level.INFO) System.out.println(); } }
	 * 
	 * public float[] predictUsingFeatureName(int featureLabelIndex, String[]
	 * wordArray) { float[] classScores = new float[this.classList.size()];
	 * float[] tempScores = new float[this.classList.size()]; for (String s:
	 * wordArray) { tempScores = this.predictUsingFeatureName(featureLabelIndex,
	 * s); for (int loop = 0; loop < tempScores.length; loop++)
	 * classScores[loop] = classScores[loop] + tempScores[loop];
	 * 
	 * 
	 * } return classScores; }
	 * 
	 * // find featureName value[], for each value[loop] get probability public
	 * float[] predictUsingFeatureName(int featureLabelIndex, String
	 * featureName) { float Pc_given_d, Pd_given_c, Pc = 0; float[] classScores
	 * = new float[this.classList.size()]; // SortedMap<String, int[]> tempMap =
	 * this.getFeatureSortedMap(featureLabelIndex); SortedMap<String, int[]>
	 * tempMap = this.labelFeatures.get(featureLabelIndex); String className =
	 * "";
	 * 
	 * if (LOGLEVEL == Level.INFO) System.out.println();
	 * 
	 * for (int loop = 0; loop < classScores.length; loop++) classScores[loop] =
	 * 0;
	 * 
	 * if (!tempMap.containsKey(featureName)) { System.out.println(featureName +
	 * " not in observed list."); } else {
	 * System.out.println("Predict using Feature Name:" + featureName); for (int
	 * loop = 0; loop < classList.size(); loop++) { className =
	 * classList.get(loop); Pd_given_c = this.getPd_given_c(classList.get(loop),
	 * featureName, tempMap); Pc = this.getPcPerLabel(classList.get(loop),
	 * tempMap); Pc_given_d = Pd_given_c * Pc; classScores[loop] = Pc_given_d;
	 * if (LOGLEVEL == Level.INFO) { System.out.print("P(" + featureName + "|" +
	 * classList.get(loop) + ")\tis " + getFeatureFreqByClass(className,
	 * featureName, tempMap) + "/" + getClassFrequencyFromFeatures(className,
	 * tempMap) + "(=" + String.format("%.3f", Pd_given_c) + ")\t* ");
	 * System.out.print("P(c)->P(" + className + ")\tis " +
	 * getClassFrequencyFromFeatures(className, tempMap) + "/" +
	 * this.getClassCountLabelFeature(className, tempMap) + "(=" +
	 * String.format("%.3f", Pc) + ")\t=  ");
	 * 
	 * System.out.println(Pc_given_d); } } } return classScores; }
	 * 
	 * // find featureName value[], for specific class public float
	 * predictUsingFeatureNameSingleClass(int classNameIndex, int
	 * featureLabelIndex, String featureName) { float Pc_given_d =0, Pd_given_c,
	 * Pc = 0; SortedMap<String, int[]> tempMap =
	 * this.labelFeatures.get(featureLabelIndex);
	 * 
	 * Pd_given_c = this.getPd_given_c(classList.get(classNameIndex),
	 * featureName, tempMap); Pc =
	 * this.getPcPerLabel(classList.get(classNameIndex), tempMap); Pc_given_d =
	 * Pd_given_c * Pc;
	 * 
	 * if (LOGLEVEL == Level.FINE) {
	 * System.out.println("Predict Class using Feature Name");
	 * System.out.print("P("
	 * +classList.get(classNameIndex)+"|"+featureName+")\t");
	 * System.out.print(" = " + String.format("%.3f", Pd_given_c) +" * ");
	 * System.out.println(String.format("%.3f",Pc) +" = " +
	 * String.format("%.3f", Pc_given_d)); } return Pc_given_d; }
	 * 
	 * //give the class scores, find larges value and return index public int
	 * returnClassScoreIndex(float[] classScores) { int foundIndex = 0; float
	 * tempFloat = 0; for (int classIndex = 0; classIndex < classScores.length;
	 * classIndex++) { if (classScores[classIndex] > tempFloat) { foundIndex =
	 * classIndex; tempFloat = classScores[classIndex]; } } return foundIndex; }
	 * 
	 * //for each class entry, determine probability of given features //
	 * assuming that newFeatures size = featureLabel size AND newFeature matches
	 * // featureLabel order public float[]
	 * predictUsingFeatureNames(List<String> checkFeatures) { float Pc_given_d =
	 * 1, Pc = 0; float[] classScores = new float[this.classList.size()];
	 * 
	 * System.out.println();
	 * System.out.println("Predict Classes using Feature Names:"); for (int
	 * clIndex = 0; clIndex < this.classList.size(); clIndex++) { // each entry
	 * equal to 1 to avoid zeroing out float Pd_given_c = 1; Pc =
	 * this.getPcPerAllFeatures(this.classList.get(clIndex), checkFeatures);
	 * System.out.print(this.classList.get(clIndex) + "\t(");
	 * 
	 * for (int cfIndex = 0; cfIndex < checkFeatures.size(); cfIndex++) {
	 * System.out.print(checkFeatures.get(cfIndex) + ":"); float
	 * local_Pd_given_c = 0; SortedMap<String, int[]> tempMap =
	 * this.labelFeatures.get(cfIndex); local_Pd_given_c = this.getPd_given_c(
	 * this.classList.get(clIndex), checkFeatures.get(cfIndex), tempMap);
	 * System.out.print(String.format("%.3f", local_Pd_given_c) + ")*(");
	 * Pd_given_c = Pd_given_c * local_Pd_given_c; }
	 * 
	 * Pc_given_d = Pd_given_c * Pc; System.out.println(String.format("%.3f",
	 * Pc) + "))\t=" + String.format("%.3f", Pc_given_d)); classScores[clIndex]
	 * = Pc_given_d; } return classScores; }
	 */

	// get total number of times className has been incremented
	private int getClassFrequencyFromFeatures(int classNameIndex, SortedMap<Float, int[]> tempMap) {
		int classFrequency = 0;
		Collection<int[]> cLoop = tempMap.values();
		Iterator<int[]> iLoop = cLoop.iterator();
		while (iLoop.hasNext()) {
			int[] iValues = iLoop.next();
			classFrequency = classFrequency + iValues[classNameIndex];
		}
		return classFrequency;
	}

	private int getClassCountLabelFeature(SortedMap<Float, int[]> tempMap) {
		int totalClasses = 0;

		// loop through all features, count number of times class has been
		// incremented
		Collection<int[]> cLoop = tempMap.values();
		Iterator<int[]> iLoop = cLoop.iterator();
		while (iLoop.hasNext()) {
			int[] iValues = iLoop.next();
			for (int loop = 0; loop < iValues.length; loop++) {
				totalClasses = totalClasses + iValues[loop];
			}
		}

		return totalClasses;
	}

	private float getFeatureKeyAtIndex(SortedMap<Float, int[]> featureMap, int index) {
		float key = 0;

		int loopCount = 0;
		for (Map.Entry<Float, int[]> entry : featureMap.entrySet()) {
			if (loopCount == index) {
				key = entry.getKey();
			}
			loopCount++;
		}
		return key;
	}

	/*
	 * private int[] getFeatureValueAtIndex(SortedMap<String, int[]> featureMap,
	 * int index) { int[] value = new int[this.classList.size()]; for (int loop
	 * = 0; loop <this.classList.size(); loop++) value[loop] = 0;
	 * 
	 * int loopCount = 0; for (Map.Entry<String, int[]> entry :
	 * featureMap.entrySet()) { if (loopCount == index) { value =
	 * entry.getValue(); } loopCount++; } return value; }
	 */
	/*
	 * private String getFeatureKeyAtIndex(SortedMap<String, int[]> featureMap,
	 * int index) { String key = "---";
	 * 
	 * int loopCount = 0; for (Map.Entry<String, int[]> entry :
	 * featureMap.entrySet()) { if (loopCount == index) { key = entry.getKey();
	 * } loopCount++; } return key; }
	 */
	private int[] getFeatureValueAtIndex(SortedMap<Float, int[]> featureMap, int index) {
		int[] value = new int[this.classList.size()];
		for (int loop = 0; loop < this.classList.size(); loop++)
			value[loop] = 0;

		int loopCount = 0;
		for (Map.Entry<Float, int[]> entry : featureMap.entrySet()) {
			if (loopCount == index) {
				value = entry.getValue();
			}
			loopCount++;
		}
		return value;
	}

	/*
	 * p(cj | d) = p(d | cj ) p(cj) ---------------- p(d)
	 */

	/*
	 * 
	 * private float getPcPerAllFeatures(String className, List<String>
	 * checkFeatures) { float Pc = 0; int classPerFeatureCount = 0; int
	 * classCountAll = 0;
	 * 
	 * 
	 * for (int lfIndex = 0; lfIndex < this.labelFeatures.size(); lfIndex++) {
	 * for (int cfIndex = 0; cfIndex < checkFeatures.size(); cfIndex++){
	 * SortedMap<String, int[]> tempMap = this.labelFeatures.get(cfIndex);
	 * classPerFeatureCount = classPerFeatureCount +
	 * this.getClassFrequencyFromFeatures(className, tempMap); classCountAll =
	 * classCountAll + this.getClassCountLabelFeature(className, tempMap); } }
	 * Pc = (float)classPerFeatureCount / classCountAll; return Pc;
	 * 
	 * }
	 * 
	 * 
	 * //P(c) // className divided by all classes private float
	 * getPcPerLabel(String className, SortedMap<String, int[]> tempMap) { float
	 * Pc; Pc = (float) getClassFrequencyFromFeatures(className, tempMap) /
	 * this.getClassCountLabelFeature(className, tempMap); return Pc; }
	 * 
	 * //find how many times a feature is associated to a class private int
	 * getFeatureFreqByClass(String className, String featureName,
	 * SortedMap<String, int[]> tempMap) { int[] classCounts =
	 * tempMap.get(featureName); return
	 * classCounts[getIndexOfClassName(className)]; }
	 * 
	 * 
	 * //Pd //featureName divided by total number of features
	 * 
	 * @SuppressWarnings("unused") private float getPd(String featureName,
	 * SortedMap<String, int[]> tempMap) { float Pd; int totalFeaturesName = 0;
	 * int totalFeatures = 0;
	 * 
	 * //loop through all features, count total number of int[] values for all
	 * features
	 * 
	 * totalFeaturesName = sumFeatureValues(tempMap.get(featureName));
	 * 
	 * //get total of all features Collection<int[]> cLoop = tempMap.values();
	 * Iterator<int[]> iLoop = cLoop.iterator();
	 * 
	 * while (iLoop.hasNext()) { totalFeatures = totalFeatures +
	 * sumFeatureValues(iLoop.next());
	 * 
	 * } Pd = (float) totalFeaturesName / totalFeatures;
	 * System.out.println("Pd --> " + featureName + ": " + totalFeaturesName +
	 * ", total # of features: " + totalFeatures+ ": " + Pd);
	 * 
	 * 
	 * return Pd; }
	 * 
	 * 
	 * 
	 * private int sumFeatureValues(int[] fValues) { int fSum = 0; for (int loop
	 * = 0; loop < fValues.length; loop++) { fSum = fSum + fValues[loop]; }
	 * return fSum; }
	 */
}
