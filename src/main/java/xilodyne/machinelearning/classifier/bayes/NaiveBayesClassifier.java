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

/**
 * Naive Bayes for classification implementation as described
 * by Prof Eamonn Keogh, UCR.
 * @see <a href="http://www.cs.ucr.edu/~eamonn/CE/Bayesian%20Classification%20withInsect_examples.pdf">http://www.cs.ucr.edu/~eamonn/CE/Bayesian%20Classification%20withInsect_examples.pdf</a> 
 * <p>
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 * 
 */
public class NaiveBayesClassifier {

	private Logger log = new Logger();

	/*
	 * Label list: list of different types of classes: name, gender, height, etc
	 * features: for each label, attributes associated to label: for label
	 * gender: male, female classificationList: which label set it be classified
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

	/** The label list. */
	private List<String> labelList = null; // different sets of features: name,
											// height,...

	/** The label features. */
	private ArrayList<SortedMap<String, int[]>> labelFeatures = null; 
	// for each label, a sorted list of each features,
	// int[] = # of classes for counting

	/**
	 * Instantiates a new naive Bayes classifier.
	 *
	 * @param newClassList
	 *            the new class list
	 * @param newLabelList
	 *            the new label list
	 */
	public NaiveBayesClassifier(List<String> newClassList, List<String> newLabelList) {
		this.classList = new ArrayList<String>();
		this.labelList = new ArrayList<String>();
		this.labelFeatures = new ArrayList<SortedMap<String, int[]>>();

		log.logln_withClassName(G.lF, "UPDATING CLASS LIST with List<String>");

		for (int loop = 0; loop < newClassList.size(); loop++)
			this.updateClasses(newClassList.get(loop));

		log.logln("UPDATING LABEL LIST");

		for (int loop = 0; loop < newLabelList.size(); loop++) {
			this.updateLabels(newLabelList.get(loop));
			this.labelFeatures.add((SortedMap<String, int[]>) new TreeMap<String, int[]>());
		}
	}

	// if a new label, add to list
	/**
	 * Update labels.
	 *
	 * @param labelName
	 *            the label name
	 */
	private void updateLabels(String labelName) {
		if (!labelList.contains(labelName)) {
			labelList.add(labelName);
			log.logln(G.lD, "LabelList[" + labelList.indexOf(labelName) + "] " + labelName);
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
			log.logln(G.lD, "ClassList[" + classList.indexOf(className) + "] " + className);
		}
	}

	/**
	 * Fit.
	 *
	 * @param featureLabelIndex
	 *            the feature label index
	 * @param featureName
	 *            the feature name
	 * @param className
	 *            the class name
	 */
	// add the training data
	public void fit(int featureLabelIndex, String featureName, String className) {
		this.updateFeatures(featureLabelIndex, featureName, this.getIndexOfClassName(className));
	}

	// assuming that newFeatures size = featureLabel size AND newFeature matches
	/**
	 * Fit.
	 *
	 * @param newFeatures
	 *            the new features
	 * @param className
	 *            the class name
	 */
	// featureLabel order
	public void fit(List<String> newFeatures, String className) {
		for (int loop = 0; loop < newFeatures.size(); loop++) {
			this.updateFeatures(loop, newFeatures.get(loop), this.getIndexOfClassName(className));
		}
	}

	/**
	 * Fit.
	 *
	 * @param featureLabelIndex
	 *            the feature label index
	 * @param observedFeatures
	 *            the observed features
	 * @param className
	 *            the class name
	 */
	// load in string array of words, associated to one class
	public void fit(int featureLabelIndex, String[] observedFeatures, String className) {
		int iIndex = this.getIndexOfClassName(className);
		for (String s : observedFeatures) {
			this.updateFeatures(featureLabelIndex, s, iIndex);
		}

	}

	/**
	 * Update features.
	 *
	 * @param featureLabelIndex
	 *            the feature label index
	 * @param featureName
	 *            the feature name
	 * @param classNameIndex
	 *            the class name index
	 */
	// add feature to the correct label, if exists, increment the classNameIndex
	private void updateFeatures(int featureLabelIndex, String featureName, int classNameIndex) {
		if (this.labelFeatures.get(featureLabelIndex).containsKey(featureName)) {
			int[] featureValues = this.labelFeatures.get(featureLabelIndex).get(featureName);

			featureValues[classNameIndex] = featureValues[classNameIndex] + 1;
			this.labelFeatures.get(featureLabelIndex).replace(featureName, featureValues);
				for (int loop = 0; loop < featureValues.length; loop++) {
					log.logln(G.lD, featureName + "[" + loop + "]:\t" + featureValues[loop]);
				}

		} else {
			int[] classCountEmpty = new int[classList.size()];
			for (int loop = 0; loop < classList.size(); loop++)
				classCountEmpty[loop] = 0;

			classCountEmpty[classNameIndex] = 1;
			this.labelFeatures.get(featureLabelIndex).put(featureName, classCountEmpty);
				for (int loop = 0; loop < classCountEmpty.length; loop++) {
					log.logln(G.lD,featureName + "[" + loop + "]:\t" + classCountEmpty[loop]);
				}
		}
	}

	/**
	 * Prints the features and classes.
	 */
	public void printFeaturesAndClasses() {
		System.out.println();
		System.out.println("Feature Frequency By Class");
		for (int labelIndex = 0; labelIndex < this.labelList.size(); labelIndex++) {
			// first line
			System.out.print(this.labelList.get(labelIndex) + "\t");

			for (int classIndex = 0; classIndex < this.classList.size(); classIndex++) {
				System.out.print(this.classList.get(classIndex) + "\t");
			}
			System.out.print("\t");
		}
		System.out.println();

		for (int featuresIndex = 0; featuresIndex < this.getLargestListFeatureSize(); featuresIndex++) {
			for (int labelFeatureIndex = 0; labelFeatureIndex < this.labelFeatures.size(); labelFeatureIndex++) {
				// for each label, show the featureName
				String featName = this.getFeatureKeyAtIndex(this.labelFeatures.get(labelFeatureIndex), featuresIndex);
				System.out.print(featName + "\t");

				int[] values = this.getFeatureValueAtIndex(this.labelFeatures.get(labelFeatureIndex), featuresIndex);
				for (int classCount = 0; classCount < values.length; classCount++) {
					System.out.print(values[classCount] + "\t");
				}
				System.out.print("\t");
			}
			System.out.println();
		}
	}

	/**
	 * Determine probabilities.
	 */
	public void determineProbabilities() {
		log.logln_noTimestamp(G.lI,"");
		log.logln("Probabilty for each feature");

			for (int labelIndex = 0; labelIndex < this.labelList.size(); labelIndex++) {
				// first line
				log.log_noTimestamp(this.labelList.get(labelIndex) + "\t");

				for (int classIndex = 0; classIndex < this.classList.size(); classIndex++) {
					log.log_noTimestamp(this.classList.get(classIndex) + "\t");
				}
				log.log_noTimestamp("\t");
			}
			log.logln_noTimestamp("");
		

		for (int featuresIndex = 0; featuresIndex < this.getLargestListFeatureSize(); featuresIndex++) {
			for (int labelFeatureIndex = 0; labelFeatureIndex < this.labelFeatures.size(); labelFeatureIndex++) {
				// for each label, show the featureName
				String featureName = this
						.getFeatureKeyAtIndex(this.labelFeatures.get(labelFeatureIndex), featuresIndex);
				log.log_noTimestamp(featureName + "\t");

				SortedMap<String, int[]> tempMap = this.labelFeatures.get(labelFeatureIndex);
				for (int classNameIndex = 0; classNameIndex < this.classList.size(); classNameIndex++) {
					float Pc_given_d = 0, Pd_given_c = 0, Pc = 0;
					// System.out.println(loopFeatures+":"+tempMap.size());
					if (tempMap.isEmpty() || ((tempMap.size() - 1) < featuresIndex)) {
						log.log_noTimestamp("\t");
					} else {
						Pd_given_c = this.getPd_given_c(classList.get(classNameIndex), featureName, tempMap);
						Pc = this.getPcPerLabel(classList.get(classNameIndex), tempMap);
						Pc_given_d = Pd_given_c * Pc;

						log.log_noTimestamp(String.format("%.3f", Pc_given_d) + "\t");
					}
				}
				// this.getProbabilities(featureLabelIndex, featureName)
				log.log_noTimestamp("\t");
			}
			log.logln_noTimestamp("");
		}
	}

	public String predict(List<String> sampleValues) {
		// return most likely class
		float[] results = this.predictUsingFeatureNames(sampleValues);
		int classIndex = this.getPredictedClass(results);
		log.logln_withClassName(G.lI, "Most likely: " + this.classList.get(classIndex));
		return this.classList.get(classIndex);
	}
	
	private int getPredictedClass(float[] results) {
		//find the greatest value
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
	
	/**
	 * Predict using feature name.
	 *
	 * @param featureLabelIndex
	 *            the feature label index
	 * @param wordArray
	 *            the word array
	 * @return the float[]
	 */
	public float[] predictUsingFeatureName(int featureLabelIndex, String[] wordArray) {
		float[] classScores = new float[this.classList.size()];
		float[] tempScores = new float[this.classList.size()];
		for (String s : wordArray) {
			tempScores = this.predictUsingFeatureName(featureLabelIndex, s);
			for (int loop = 0; loop < tempScores.length; loop++)
				classScores[loop] = classScores[loop] + tempScores[loop];

		}
		return classScores;
	}

	/**
	 * Predict using feature name.
	 *
	 * @param featureLabelIndex
	 *            the feature label index
	 * @param featureName
	 *            the feature name
	 * @return the float[]
	 */
	// find featureName value[], for each value[loop] get probability
	public float[] predictUsingFeatureName(int featureLabelIndex, String featureName) {
		float Pc_given_d, Pd_given_c, Pc = 0;
		float[] classScores = new float[this.classList.size()];
		// SortedMap<String, int[]> tempMap =
		// this.getFeatureSortedMap(featureLabelIndex);
		SortedMap<String, int[]> tempMap = this.labelFeatures.get(featureLabelIndex);
		String className = "";


		for (int loop = 0; loop < classScores.length; loop++)
			classScores[loop] = 0;

		if (!tempMap.containsKey(featureName)) {
			System.out.println(featureName + " not in observed list.");
		} else {
			log.logln_noTimestamp(G.lI, "");
			log.logln("Predict using Feature Name:" + featureName);
			for (int loop = 0; loop < classList.size(); loop++) {
				className = classList.get(loop);
				Pd_given_c = this.getPd_given_c(classList.get(loop), featureName, tempMap);
				Pc = this.getPcPerLabel(classList.get(loop), tempMap);
				Pc_given_d = Pd_given_c * Pc;
				classScores[loop] = Pc_given_d;
				log.log_noTimestamp("P(" + featureName + "|" + classList.get(loop) + ")\tis "
							+ getFeatureFreqByClass(className, featureName, tempMap) + "/"
							+ getClassFrequencyFromFeatures(className, tempMap) + "(="
							+ String.format("%.3f", Pd_given_c) + ")\t* ");
				log.log_noTimestamp("P(c)->P(" + className + ")\tis "
							+ getClassFrequencyFromFeatures(className, tempMap) + "/"
							+ this.getClassCountLabelFeature(className, tempMap) + "(=" + String.format("%.3f", Pc)
							+ ")\t=  ");

				log.logln_noTimestamp(String.valueOf(Pc_given_d));
			}
		}
		return classScores;
	}

	/**
	 * Predict using feature name single class.
	 *
	 * @param classNameIndex
	 *            the class name index
	 * @param featureLabelIndex
	 *            the feature label index
	 * @param featureName
	 *            the feature name
	 * @return the float
	 */
	// find featureName value[], for specific class
	public float predictUsingFeatureNameSingleClass(int classNameIndex, int featureLabelIndex, String featureName) {
		float Pc_given_d = 0, Pd_given_c, Pc = 0;
		SortedMap<String, int[]> tempMap = this.labelFeatures.get(featureLabelIndex);

		Pd_given_c = this.getPd_given_c(classList.get(classNameIndex), featureName, tempMap);
		Pc = this.getPcPerLabel(classList.get(classNameIndex), tempMap);
		Pc_given_d = Pd_given_c * Pc;
		
		log.logln_noTimestamp(G.lI, "");

		log.logln("Predict Class using Feature Name Single Class");
			log.log_noTimestamp("P(" + classList.get(classNameIndex) + "|" + featureName + ")\t");
			log.log_noTimestamp(" = " + String.format("%.3f", Pd_given_c) + " * ");
			log.logln_noTimestamp(String.format("%.3f", Pc) + " = " + String.format("%.3f", Pc_given_d));
		
		return Pc_given_d;
	}

	/**
	 * Return class score index.
	 *
	 * @param classScores
	 *            the class scores
	 * @return the int
	 */
	// give the class scores, find larges value and return index
	public int returnClassScoreIndex(float[] classScores) {
		int foundIndex = 0;
		float tempFloat = 0;
		for (int classIndex = 0; classIndex < classScores.length; classIndex++) {
			if (classScores[classIndex] > tempFloat) {
				foundIndex = classIndex;
				tempFloat = classScores[classIndex];
			}
		}
		return foundIndex;
	}

	// for each class entry, determine probability of given features
	// assuming that newFeatures size = featureLabel size AND newFeature matches
	/**
	 * Predict using feature names.
	 *
	 * @param checkFeatures
	 *            the check features
	 * @return the float[]
	 */
	// featureLabel order
	public float[] predictUsingFeatureNames(List<String> checkFeatures) {
		float Pc_given_d = 1, Pc = 0;
		float[] classScores = new float[this.classList.size()];

		log.logln_noTimestamp(G.lI, "");
		log.logln("Predict Classes using Feature Names:");
		for (int classListIndex = 0; classListIndex < this.classList.size(); classListIndex++) {
			// each entry equal to 1 to avoid zeroing out
			float Pd_given_c = 1;
			Pc = this.getPcPerAllFeatures(this.classList.get(classListIndex), checkFeatures);
			log.log_noTimestamp(this.classList.get(classListIndex) + "\t(");

			for (int checkFeatIndex = 0; checkFeatIndex < checkFeatures.size(); checkFeatIndex++) {
				log.log_noTimestamp(checkFeatures.get(checkFeatIndex) + ":");
				float local_Pd_given_c = 0;
				SortedMap<String, int[]> tempMap = this.labelFeatures.get(checkFeatIndex);
				local_Pd_given_c = this.getPd_given_c(this.classList.get(classListIndex),
						checkFeatures.get(checkFeatIndex), tempMap);
				log.log_noTimestamp(String.format("%.3f", local_Pd_given_c) + ")*(");
				Pd_given_c = Pd_given_c * local_Pd_given_c;
			}

			Pc_given_d = Pd_given_c * Pc;
			log.logln_noTimestamp(String.format("%.3f", Pc) + "))\t=" + String.format("%.3f", Pc_given_d));
			classScores[classListIndex] = Pc_given_d;
		}
		return classScores;
	}

	// private SortedMap<String, int[]> getFeatureSortedMap(int
	// listFeatureIndex){
	// SortedMap<String, int[]> tempMap =
	// this.labelFeatures.get(listFeatureIndex);
	// System.out.println(this.labelFeatures.size()+":"+listFeatureIndex
	// +":"+tempMap.firstKey());
	// return this.labelFeatures.get(listFeatureIndex);
	// }

	/**
	 * Gets the feature key at index.
	 *
	 * @param featureMap
	 *            the feature map
	 * @param index
	 *            the index
	 * @return the feature key at index
	 */
	private String getFeatureKeyAtIndex(SortedMap<String, int[]> featureMap, int index) {
		String key = "---";

		int loopCount = 0;
		for (Map.Entry<String, int[]> entry : featureMap.entrySet()) {
			if (loopCount == index) {
				key = entry.getKey();
			}
			loopCount++;
		}
		return key;
	}

	/**
	 * Gets the feature value at index.
	 *
	 * @param featureMap
	 *            the feature map
	 * @param index
	 *            the index
	 * @return the feature value at index
	 */
	private int[] getFeatureValueAtIndex(SortedMap<String, int[]> featureMap, int index) {
		int[] value = new int[this.classList.size()];
		for (int loop = 0; loop < this.classList.size(); loop++)
			value[loop] = 0;

		int loopCount = 0;
		for (Map.Entry<String, int[]> entry : featureMap.entrySet()) {
			if (loopCount == index) {
				value = entry.getValue();
			}
			loopCount++;
		}
		return value;
	}

	/**
	 * Gets the largest list feature size.
	 *
	 * @return the largest list feature size
	 */
	private int getLargestListFeatureSize() {
		int largeIndex = 0;
		for (int loop = 0; loop < this.labelFeatures.size(); loop++) {
			if (largeIndex < this.labelFeatures.get(loop).size()) {
				largeIndex = this.labelFeatures.get(loop).size();
			}
		}
		return largeIndex;
	}

	/*
	 * p(cj | d) = p(d | cj ) p(cj) ---------------- p(d)
	 */

	/**
	 * Gets the pd given c.
	 *
	 * @param className
	 *            the class name
	 * @param featureName
	 *            the feature name
	 * @param tempMap
	 *            the temp map
	 * @return the pd given c
	 */
	public float getPd_given_c(String className, String featureName, SortedMap<String, int[]> tempMap) {
		float Pd_given_c = 0;
		// p(d | cj )
		// given class, determine number of times featureName has className /
		// total # className
		if (tempMap.containsKey(featureName)) {
			Pd_given_c = (float) getFeatureFreqByClass(className, featureName, tempMap)
					/ getClassFrequencyFromFeatures(className, tempMap);

		} else {
			System.out.println("No instances of " + featureName + ".");
			try {
				throw new Exception("No instances of " + featureName + ".");
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		// System.out.println("(" + className + "|" + featureName + ") is " +
		// ((float) Pd_given_c * getPc(className)));

		return Pd_given_c;
	}

	/**
	 * Gets the pc per all features.
	 *
	 * @param className
	 *            the class name
	 * @param checkFeatures
	 *            the check features
	 * @return the pc per all features
	 */
	private float getPcPerAllFeatures(String className, List<String> checkFeatures) {
		float Pc = 0;
		int classPerFeatureCount = 0;
		int classCountAll = 0;

		for (int lfIndex = 0; lfIndex < this.labelFeatures.size(); lfIndex++) {
			for (int cfIndex = 0; cfIndex < checkFeatures.size(); cfIndex++) {
				SortedMap<String, int[]> tempMap = this.labelFeatures.get(cfIndex);
				classPerFeatureCount = classPerFeatureCount + this.getClassFrequencyFromFeatures(className, tempMap);
				classCountAll = classCountAll + this.getClassCountLabelFeature(className, tempMap);
			}
		}
		Pc = (float) classPerFeatureCount / classCountAll;
		return Pc;

	}

	// P(c)
	/**
	 * Gets the pc per label.
	 *
	 * @param className
	 *            the class name
	 * @param tempMap
	 *            the temp map
	 * @return the pc per label
	 */
	// className divided by all classes
	private float getPcPerLabel(String className, SortedMap<String, int[]> tempMap) {
		float Pc;
		Pc = (float) getClassFrequencyFromFeatures(className, tempMap)
				/ this.getClassCountLabelFeature(className, tempMap);
		return Pc;
	}

	/**
	 * Gets the feature freq by class.
	 *
	 * @param className
	 *            the class name
	 * @param featureName
	 *            the feature name
	 * @param tempMap
	 *            the temp map
	 * @return the feature freq by class
	 */
	// find how many times a feature is associated to a class
	private int getFeatureFreqByClass(String className, String featureName, SortedMap<String, int[]> tempMap) {
		int[] classCounts = tempMap.get(featureName);
		return classCounts[getIndexOfClassName(className)];
	}

	/**
	 * Gets the class count label feature.
	 *
	 * @param className
	 *            the class name
	 * @param tempMap
	 *            the temp map
	 * @return the class count label feature
	 */
	private int getClassCountLabelFeature(String className, SortedMap<String, int[]> tempMap) {
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

	/**
	 * Gets the class frequency from features.
	 *
	 * @param className
	 *            the class name
	 * @param tempMap
	 *            the temp map
	 * @return the class frequency from features
	 */
	// get total number of times className has been incremented
	private int getClassFrequencyFromFeatures(String className, SortedMap<String, int[]> tempMap) {
		int classFrequency = 0;
		Collection<int[]> cLoop = tempMap.values();
		Iterator<int[]> iLoop = cLoop.iterator();
		while (iLoop.hasNext()) {
			int[] iValues = iLoop.next();
			classFrequency = classFrequency + iValues[getIndexOfClassName(className)];
		}
		return classFrequency;
	}

	// Pd
	/**
	 * Gets the pd.
	 *
	 * @param featureName
	 *            the feature name
	 * @param tempMap
	 *            the temp map
	 * @return the pd
	 */
	// featureName divided by total number of features
	@SuppressWarnings("unused")
	private float getPd(String featureName, SortedMap<String, int[]> tempMap) {
		float Pd;
		int totalFeaturesName = 0;
		int totalFeatures = 0;

		// loop through all features, count total number of int[] values for all
		// features

		totalFeaturesName = sumFeatureValues(tempMap.get(featureName));

		// get total of all features
		Collection<int[]> cLoop = tempMap.values();
		Iterator<int[]> iLoop = cLoop.iterator();

		while (iLoop.hasNext()) {
			totalFeatures = totalFeatures + sumFeatureValues(iLoop.next());

		}
		Pd = (float) totalFeaturesName / totalFeatures;
		System.out.println("Pd --> " + featureName + ": " + totalFeaturesName + ", total # of features: "
				+ totalFeatures + ": " + Pd);

		return Pd;
	}

	/**
	 * Gets the index of class name.
	 *
	 * @param className
	 *            the class name
	 * @return the index of class name
	 */
	private int getIndexOfClassName(String className) {
		if (!classList.contains(className))
			System.out.println("*** ERROR: " + className + " not in list. ***");
		return classList.indexOf(className);

	}

	/**
	 * Sum feature values.
	 *
	 * @param fValues
	 *            the f values
	 * @return the int
	 */
	private int sumFeatureValues(int[] fValues) {
		int fSum = 0;
		for (int loop = 0; loop < fValues.length; loop++) {
			fSum = fSum + fValues[loop];
		}
		return fSum;
	}

}
