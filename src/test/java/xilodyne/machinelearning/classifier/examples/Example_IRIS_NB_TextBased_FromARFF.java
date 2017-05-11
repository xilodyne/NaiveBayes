package xilodyne.machinelearning.classifier.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;

import mikera.arrayz.NDArray;
import weka.core.Instances;
import xilodyne.machinelearning.classifier.bayes.NaiveBayesClassifier_UsingTextValues;
import xilodyne.util.ArrayUtils;
import xilodyne.util.io.FileSplitter;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.LoggerCSV;
import xilodyne.util.metrics.OutputResults;
import xilodyne.util.metrics.TestResultsDataML;
import xilodyne.util.weka.WekaARFFUtils;
import xilodyne.util.weka.WekaUtils;

/**
 * Naive Bayes using Iris Data Set.
 * https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
 * 
 * Uses NDArray by vectorz https://github.com/mikera/vectorz
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.2
 * 
 */
public class Example_IRIS_NB_TextBased_FromARFF {

	public static String CSV_Filename = "IRIS_Data_NaiveBayes-XD";
	public static String delimiter = ",";
	public static String[] header = {"timestamp", "class name", 
		"accuracy", "CV Fold", "# of lines", "# trained", "date time", 
		"train time", "predict time","total time"};

	private static Logger log;
	private static LoggerCSV logCSV;

	//private static Instant startData, endData, startFit, endFit, startPredict, endPredict = null;
	
	private static ArrayList<String> predictResults = null;
	private static ArrayList<String> labels = null;
	
	private static int indexClassLabel;
	


	// get metadata,
	// open file,
	// get number of columns
	// get number of rows
	// assume last column are class
	static NDArray dataArray = null;
	static double[] labeledData = null;
	
	private static NaiveBayesClassifier_UsingTextValues nb = new NaiveBayesClassifier_UsingTextValues(NaiveBayesClassifier_UsingTextValues.EMPTY_SAMPLES_IGNORE);
	static String[] sLabels = null;
	static String[] sFeatures = null;

	public static void main(String[] args) {
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		G.setLoggerLevel(G.LOG_DEBUG);

		String classNamex = "xilodyne.machinelearning.classifier.NaiveBayesClassifier";
		TestResultsDataML resultsData = new TestResultsDataML();
		resultsData.setClassMLName(classNamex);
		
	//	log = new Logger("logs", "IRIS_xd_NB_NoCV" + "_" + className.substring(className.lastIndexOf(".") + 1));
		log = new Logger("logs", "IRIS_xd_NB_NoCV" + "_" + resultsData.getClassMLNameWithoutDomain());
		logCSV = new LoggerCSV("results", CSV_Filename, 
				delimiter, header);

		log.logln_withClassName(G.lF,"");
		
		log.logln("IRIS data set using Naive Bayes");
		logCSV.log_CSV_Timestamp();
		//logCSV.log_CSV_Entry(className);
		logCSV.log_CSV_Entry(resultsData.getClassMLName());

		//startData = Instant.now();
		resultsData.setStartData();
		
		String filePath = "./test-data";
		String fileName = "iris.arff";
		indexClassLabel = 4;  //4 data values, 1 label value)
		Instances data = null;

			//split the arff file into info and data, 
			//split the data file into 5 pieces for training and testing
			//randomize file as arff file is sorted by label

			//FileSplitter.createSubFilesFromARFF(5, filePath, fileName, FileSplitter.fileExtARFF);
			FileSplitter.createSubARFF_Shuffle(10, filePath, fileName, FileSplitter.fileExtARFF, FileSplitter.SHUFFLE);
			//using weka, get @ATTRIBUTES from info file
		//	Instances data = WekaARFFUtils.wekaReadARFF(filePath + "/" +
		//			fileName + "." + FileSplitter.fileExtARFF_INFO);
			data = WekaARFFUtils.wekaReadARFF(filePath + "/" +
					fileName);
			
			WekaUtils.printInstanceDetails("Iris", data, log);
			//log.logln(ArrayUtils.printArray(
					//WekaUtils.getClassNames(data, WekaUtils.ClassAtEnd)));
			



		sLabels = WekaUtils.getLabelNames(data);
		sFeatures = WekaUtils.getFeatureNames(data);
		log.logln(G.lD, ArrayUtils.printArray(sLabels));

		List<String> labelNames = new ArrayList<String>(Arrays.asList(sLabels));
		List<String> featureNames = new ArrayList<String>(Arrays.asList(sFeatures));
		log.logln(ArrayUtils.printArray(labelNames));		
		log.logln(ArrayUtils.printArray(featureNames));

		//endData = Instant.now();
		resultsData.setEndData();
		
		//startFit = Instant.now();
		resultsData.setStartFit();
		try {
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 1,indexClassLabel);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 2,indexClassLabel);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 3,indexClassLabel);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 4,indexClassLabel);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 5,indexClassLabel);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 6,indexClassLabel);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 7,indexClassLabel);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 8,indexClassLabel);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 9,indexClassLabel);
		} catch (IOException e) {

			e.printStackTrace();
		}

		//endFit = Instant.now();
		resultsData.setEndFit();
	//	nb.printFeaturesAndClasses();


		//startPredict = Instant.now();
		resultsData.setStartPredict();
		try {
		//iris-versicolor
			//float[] fval = nb.predictUsingFeatureNames(new ArrayList<String>(Arrays.asList("5.4","3.0","4.5","1.5")));
			//log.logln ("result: " + ArrayUtils.printArray(fval));
			//log.logln(nb.predict(new ArrayList<String>(Arrays.asList("5.4","3.0","4.5","1.5"))));
			predictResults = predict(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 10,indexClassLabel);
			log.logln(G.lI, "Predicted Count: " + predictResults.size());

		} catch (Exception e) {
			e.printStackTrace();
		}
		//endPredict = Instant.now();
		resultsData.setEndPredict();

		try {
		labels = getLabelsFromFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 10,indexClassLabel);
	} catch (Exception e) {
		e.printStackTrace();
	}


		
		resultsData.setAccuracy(ArrayUtils.getAccuracyOfLabels(predictResults,  labels) * 100);
		resultsData.setTrainingDataSize(nb.getFitCount());
		resultsData.setTestingDataSize(predictResults.size());
		OutputResults.getMLStats(log, logCSV, resultsData);
		logCSV.log_CSV_EOL();

	}


	// given file, load data minus label
	private static void fitDataFile(String filePath, String fileName, 
			int fileNumber, int indexOfLabel) throws IOException {

		String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;

		log.logln("Loading file " + file + ",");

		//for each line, load data
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			log.logln(G.lD, "Row: " + ArrayUtils.printArray(values));
			
			//nb.fit(new ArrayList<String>(Arrays.asList("Drew", "No", "Blue", "Short")), "Male");

			String label = null;
			// load the last value into the class array
			log.log("Values extracted [");
			//get the label first
			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
					label = values[index];
					log.logln("Label: " + label);
				} 
			}
			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
				//	label = values[index];
				} else {
					//list.add(values[index]);		
					nb.fit(sFeatures[index], values[index], label);
				}
			}
			
			log.logln_noTimestamp("]");

		//	log.logln(G.lD,  "Loading : " + ArrayUtils.printArray(list));

		//	nb.fit(list, label);

		}
		br.close();
	}
	
	// given file, load data minus label
	//return a list of results
	private static ArrayList<String> predict(String filePath, String fileName, 
			int fileNumber, int indexOfLabel) throws IOException {

		String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;
		ArrayList<String> labels = new ArrayList<String>();

		log.logln("Loading file " + file + ",");

		//for each line, load data
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			log.logln(G.lI, "");
			log.logln(G.lD, "Row: " + ArrayUtils.printArray(values));
			
			//nb.fit(new ArrayList<String>(Arrays.asList("Drew", "No", "Blue", "Short")), "Male");

			
			// load the last value into the class array
			log.log(G.lI, "Values extracted [");
			Hashtable<String, String> testingData_OneSet = new Hashtable<String, String>();

			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
		//			labels.add(values[index]);
				} else {
				//	list.add(values[index]);	
					testingData_OneSet.put(sFeatures[index], values[index]);
				}
			}
			log.logln_noTimestamp("]");

		//	log.logln(G.lF, predictedCount +":" + labels.size() +": Label: " + labels.get(labels.size()-1));
			String predictedLabel = nb.predict_TestingSet(testingData_OneSet);
			labels.add(predictedLabel);
			log.logln(G.lI, "\nLooking at: " + ArrayUtils.printArray(testingData_OneSet));
			log.logln("Predicted: " + predictedLabel);
		}
		br.close();
		return labels;
	}
	
	// given file, load data minus label
	//return a list of results
	private static ArrayList<String> getLabelsFromFile(String filePath, String fileName, 
			int fileNumber, int indexOfLabel) throws IOException {

		String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;
		ArrayList<String> labels = new ArrayList<String>();

		log.logln("Loading file " + file + ",");

		//for each line, load data
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			log.logln(G.lD, "");
			log.logln(G.lD, "Row: " + ArrayUtils.printArray(values));
			
			//nb.fit(new ArrayList<String>(Arrays.asList("Drew", "No", "Blue", "Short")), "Male");

			//ArrayList<String> list = new ArrayList<String>();
			
			// load the last value into the class array
			log.log(G.lD, "Values extracted [");
			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
					labels.add(values[index]);
				} 
			}
			log.logln_noTimestamp("]");

			log.logln(G.lD, "Label: " + labels.get(labels.size()-1));

		}
		br.close();
		return labels;
	}
	
/*	private static void getStatsOld(String className) {
		accuracy = (long) (ArrayUtils.getAccuracyOfLabels(predictResults,  labels) * 100);
		long dataTime = Duration.between(startData, endData).toMillis();
		long trainingTime = Duration.between(startFit, endFit).toMillis();
		long predictTime = Duration.between(startPredict, endPredict).toMillis();
		double dDataTime = dataTime / (double) 1000;
		double dTrainTime = trainingTime / (double) 1000;
		double dPredictTime = predictTime / (double) 1000;

		double totalDuration = dDataTime + dTrainTime + dPredictTime;
		long dataPercent = Math.round((dDataTime / totalDuration) * 100);
		long trainPercent = Math.round((dTrainTime / totalDuration) * 100);
		long predictPercent = Math.round((dPredictTime / totalDuration) * 100);
		log.logln(G.lF, "Class tested: " + className);
		log.logln("Accuracy: " + accuracy + "%");
		log.logln("Total lines training: " + nb.getFitCount());
		log.logln("Total lines predicted: " + predictResults.size());
		// log.logln("Training time: " + trainingTime + " milliseconds.");
		// log.logln("Predict time: " + predictTime + " milliseconds.");
		log.logln("Activity\tTime (in seconds)\t% of Total Duration");
		log.logln("--------\t-----------------\t-------------------");
		// log.logln("Data setup\t" + dDataTime + "\t" +
		// System.out.format("%fn", dataPercent));
		log.logln("Data setup\t" + dDataTime + "\t\t\t" + dataPercent + "%");
		log.logln("Training\t" + dTrainTime + "\t\t\t" + trainPercent + "%");
		log.logln("Predict\t\t" + dPredictTime + "\t\t\t" + predictPercent + "%");

	}
	
	public static void getStats(String className) {
		accuracy = ArrayUtils.getAccuracyOfLabels(predictResults,  labels) * 100;

		long dataTime = Duration.between(startData, endData).toMillis();
		long trainingTime = Duration.between(startFit, endFit).toMillis();
		long predictTime = Duration.between(startPredict, endPredict).toMillis();
		double dDataTime = dataTime / (double) 1000;
		double dTrainTime = trainingTime / (double) 1000;
		double dPredictTime = predictTime / (double) 1000;

		double totalDuration = dDataTime + dTrainTime + dPredictTime;
		long dataPercent = Math.round((dDataTime / totalDuration) * 100);
		long trainPercent = Math.round((dTrainTime / totalDuration) * 100);
		long predictPercent = Math.round((dPredictTime / totalDuration) * 100);
		log.logln(G.lF, "Class tested: " + className);
		log.logln("Accuracy: " + accuracy + "%");
		log.logln("Total lines training: " + nb.getFitCount());
		log.logln("Total lines predicted: " + predictResults.size());
		// log.logln("Training time: " + trainingTime + " milliseconds.");
		// log.logln("Predict time: " + predictTime + " milliseconds.");
		log.logln("Activity\tTime (in seconds)\t% of Total Duration");
		log.logln("--------\t-----------------\t-------------------");
		// log.logln("Data setup\t" + dDataTime + "\t" +
		// System.out.format("%fn", dataPercent));
		log.logln("Data setup\t" + dDataTime + "\t\t\t" + dataPercent + "%");
		log.logln("Training\t" + dTrainTime + "\t\t\t" + trainPercent + "%");
		log.logln("Predict\t\t" + dPredictTime + "\t\t\t" + predictPercent + "%" );
		log.logln("Total Time\t" + totalDuration);
		logCSV.log_CSV_Entry(String.valueOf(accuracy));
		logCSV.log_CSV_Entry("-");
		logCSV.log_CSV_Entry(String.valueOf(nb.getFitCount()));
		logCSV.log_CSV_Entry(String.valueOf(predictResults.size()));
		logCSV.log_CSV_Entry(String.valueOf(dDataTime));
		logCSV.log_CSV_Entry(String.valueOf(dTrainTime));
		logCSV.log_CSV_Entry(String.valueOf(dPredictTime));
		logCSV.log_CSV_Entry(String.valueOf(totalDuration));


		// double acc = (double)
		// ArrayUtils.getNumberOfCorrectMatches(predictResults,
		// labels)/predictResults.size();
		// System.out.println("Accuracy: " +
		// ArrayUtils.getAccuracyOfLabels(predictResults, labels));

	}
*/

}
