package xilodyne.machinelearning.classifier.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mikera.arrayz.NDArray;
import xilodyne.machinelearning.classifier.bayes.GaussianNaiveBayesClassifier;
import xilodyne.util.ArrayUtils;
import xilodyne.util.file.io.FileSplitter;
import xilodyne.util.metrics.OutputResults;
import xilodyne.util.metrics.TestResultsDataML;
import xilodyne.util.logger.Logger;
import xilodyne.util.logger.LoggerCSV;
import xilodyne.util.weka_helper.WekaARFFUtils;
import xilodyne.util.weka_helper.WekaUtils;
import weka.core.Instances;

/**
 * Gaussian NB using Iris Data Set.
 * https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
 * 
 * Uses NDArray by vectorz https://github.com/mikera/vectorz
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.4 - 1/30/2018 - reflect xilodyne util changes
 * @version 0.2
 * 
 */
public class Example_IRIS_GNB_FromARFF {
	public static String CSV_Filename = "IRIS_Data_GaussianNaiveBayes-XD";
	public static String delimiter = ",";
	public static String[] header = {"timestamp", "class name", 
		"accuracy", "CV Fold", "# of lines", "# trained", "date time", 
		"train time", "predict time","total time"};

	private static Logger log;
	private static LoggerCSV logCSV;

	private static int indexClassLabel, numOfFeatures;
	
	private static double[] predictResults, labels;

	private static GaussianNaiveBayesClassifier gnb = null;
	// get metadata,
	// open file,
	// get number of columns
	// get number of rows
	// assume last column are class
	

	public static void main(String[] args) {

		// Logger.setLoggerLevel(Logger.LOG_OFF);
		// Logger.setLoggerLevel(Logger.LOG_FINE);
		// Logger.setLoggerLevel(Logger.LOG_INFO);
		Logger.setLoggerLevel(Logger.LOG_DEBUG);

		String className = "xilodyne.machinelearning.classifier.GaussianNB";
		TestResultsDataML resultsData = new TestResultsDataML();
		resultsData.setClassMLName(className);

//		log = new Logger("logs", "IRIS_xd_GNB_NoCV" + "_" + className.substring(className.lastIndexOf(".") + 1));
		log = new Logger("egnb", "logs", "IRIS_xd_GNB_NoCV" + "_" + resultsData.getClassMLNameWithoutDomain());

		logCSV = new LoggerCSV("results", CSV_Filename, 
				delimiter, header);
		logCSV.log_CSV_Timestamp();
		logCSV.log_CSV_Entry(resultsData.getClassMLName());

		
		log.logln_withClassName(Logger.lF,"");

		//startData = Instant.now();
		resultsData.setStartData();
		
		String filePath = "./test-data";
		String fileName = "iris.arff";
		indexClassLabel = 4;  //4 data values, 1 label value)
		numOfFeatures = 4;
		try {
			//split the arff file into info and data, 
			//split the data file into 5 pieces for training and testing
			//randomize file as arff file is sorted by label
			FileSplitter.createSubARFF_Shuffle(10, filePath, fileName, FileSplitter.fileExtARFF, FileSplitter.SHUFFLE);

			//using weka, get @ATTRIBUTES from info file
		//	Instances data = WekaARFFUtils.wekaReadARFF(filePath + "/" +
		//			fileName + "." + FileSplitter.fileExtARFF_INFO);
			Instances data = WekaARFFUtils.wekaReadARFF(filePath + "/" +
					fileName);
			
			//WekaUtils.printInstanceDetails("Iris", data, log);
			//System.out.println(ArrayUtils.printArray(
					//WekaUtils.getClassNames(data, WekaUtils.ClassAtEnd)));
			String[] sLabels = WekaUtils.getLabelNames(data);
			String[] sFeatures = WekaUtils.getFeatureNames(data);
			log.logln(Logger.lD, ArrayUtils.printArray(sLabels));

			ArrayList<String> labelNames = new ArrayList<String>(Arrays.asList(sLabels));
			List<String> featureNames = new ArrayList<String>(Arrays.asList(sFeatures));
			log.logln(ArrayUtils.printArray(labelNames));		
			log.logln(ArrayUtils.printArray(featureNames));

			//endData = Instant.now();
			resultsData.setEndData();
			
			gnb = new GaussianNaiveBayesClassifier(GaussianNaiveBayesClassifier.EMPTY_SAMPLES_IGNORE);

			//startFit = Instant.now();
			resultsData.setStartFit();

			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 1, indexClassLabel, numOfFeatures, labelNames);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 2, indexClassLabel, numOfFeatures, labelNames);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 3, indexClassLabel, numOfFeatures, labelNames);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 4, indexClassLabel, numOfFeatures, labelNames);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 5, indexClassLabel, numOfFeatures, labelNames);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 6, indexClassLabel, numOfFeatures, labelNames);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 7, indexClassLabel, numOfFeatures, labelNames);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 8, indexClassLabel, numOfFeatures, labelNames);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 9, indexClassLabel, numOfFeatures, labelNames);
			
			//endFit = Instant.now();
			resultsData.setEndFit();

			//startPredict = Instant.now();
			resultsData.setStartPredict();
			predictResults = predict(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 10, indexClassLabel, numOfFeatures, labelNames);
			//endPredict = Instant.now();
			resultsData.setEndPredict();

			labels = getLabelsFromfile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 10, indexClassLabel, labelNames);
			log.logln(Logger.lF, "Predicted Results size: " + predictResults.length);
			log.logln(Logger.lF, "Class Labels size: " + labels.length);
			
			/*long trainingTime = Duration.between(startFit, endFit).toMillis();
			long predictTime = Duration.between(startPredict, endPredict).toMillis();
			System.out.println("Total lines loaded: " + gnb.getFitCount());
			System.out.println("Total lines predicted: " + predictResults.length);
			System.out.println("Training time GNB: " + trainingTime + " milliseconds.");
			System.out.println("Predict time GNB: " + predictTime + " milliseconds.");
			double acc = (double) ArrayUtils.getNumberOfCorrectMatches(predictResults,  labels)/predictResults.length;
			System.out.println("Matched " + ArrayUtils.getNumberOfCorrectMatches(predictResults,  labels) + " out of " + predictResults.length + " entries.");
			System.out.println("Accuracy: "  + ArrayUtils.getAccuracyOfLabels(predictResults,  labels));
*/
			
//			private static void fitDataFile(String filePath, String fileName, 
					//int fileNumber, int numOfFeatures, int indexOfLabel, ArrayList<String> labelNames) throws IOException {

				
			//load2D_NDArray(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 5, 9);

			//log.logln_withClassName(Logger.lI, "Output of ND Array...");
			//log.logln("\ndata: "+ dataArray);
			//log.logln("ND array dim: " + dataArray.getShape(1));
			
			//printlabeledData();

	//		System.out.println(WekaUtils.getInstanceDetails(data));
	/*		load2D_NDArray(filePath, fileName, 5, 5);

			log.logln_withClassName(Logger.lD, "Output of ND Array...");
			log.logln("\ndata: "+ dataArray);
			log.logln("ND array dim: " + dataArray.getShape(1));
			
			// printlabeledData();


			gnb.fit(dataArray, labeledData);

			load2D_NDArray(filePath, fileName, 4, 5);

			log.logln(Logger.lD, "Output of ND Array...");
			log.logln(dataArray.toString());
			log.logln("ND array size: " + dataArray.getShape(1));
			
			// printlabeledData();

			gnb.fit(dataArray, labeledData);

			load2D_NDArray(filePath, fileName, 1, 5);

			double[] predictedResults = gnb.predict(dataArray);
			log.logln("Predicted Results size: " + predictedResults.length);
			log.logln("Class Labels size: " + labeledData.length);
			

			double accuracy = gnb.getAccuracyOfPredictedResults(labeledData, predictedResults);
			System.out.println("Accuracy: " + accuracy);
*/
	//		getStats(className);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		resultsData.setAccuracy(ArrayUtils.getAccuracyOfLabels(predictResults,  labels) * 100);
		resultsData.setTrainingDataSize(gnb.getFitCount());
		resultsData.setTestingDataSize(predictResults.length);
		OutputResults.getMLStats(log, logCSV, resultsData);
		logCSV.log_CSV_EOL();

	}

	// load the data into the ndarray
	// assuming the last value of the line is the labeled Class value
	//as label names in the ARFF file are strings, need to convert to ints
	// given file, load data minus label
	private static void fitDataFile(String filePath, String fileName, 
			int fileNumber, int numOfFeatures, int indexOfLabel, ArrayList<String> labelNames) throws IOException {

		String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);
		int numOfLines = FileSplitter.getLineCount(filePath,
				FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA));
	
		NDArray dataArray = NDArray.newArray(numOfLines, (numOfFeatures));
		double[] labels = new double[numOfLines];

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;

		log.logln("Loading file " + file + ",");
		int lineNum = 0;

		//for each line, load data
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			//log.logln(Logger.lD, "Row: " + ArrayUtils.printArray(values));
			
			String label = null;
			// load the last value into the class array
			log.log(lineNum + ": Loading [");
			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
					label = values[index];
					labels[lineNum] = labelNames.indexOf(label);
					log.log_noTimestamp(" label: " + label + ", index: " + labels[index]);
				} else {
					log.log_noTimestamp(" (" + index +":" + values[index] +") ");
					dataArray.set(lineNum, index, Double.valueOf(values[index]));
				}
			}
			lineNum++;
			log.logln_noTimestamp("]");
//			nb.fit(list, label);

		}

		log.logln(Logger.lD,  "Loading : " + ArrayUtils.printArray(labels));
		log.logln("Data: " + dataArray);
		br.close();
		try {
			gnb.fit(dataArray,  labels);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static double[] predict(String filePath, String fileName, 
			int fileNumber, int numOfFeatures, int indexOfLabel, ArrayList<String> labelNames) throws IOException {

		double[] predResults = null;
		
		String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);
		int numOfLines = FileSplitter.getLineCount(filePath,
				FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA));
	
		NDArray dataArray = NDArray.newArray(numOfLines, (numOfFeatures));
		double[] labels = new double[numOfLines];

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;

		log.logln("Loading file " + file + ",");
		int lineNum = 0;

		//for each line, load data
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			//log.logln(Logger.lD, "Row: " + ArrayUtils.printArray(values));
			
			String label = null;
			// load the last value into the class array
			log.log(lineNum + ": Loading [");
			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
					label = values[index];
					labels[lineNum] = labelNames.indexOf(label);
					log.log_noTimestamp(" label: " + label + ", index: " + labels[index]);
				} else {
					log.log_noTimestamp(" (" + index +":" + values[index] +") ");
					dataArray.set(lineNum, index, Double.valueOf(values[index]));
				}
			}
			lineNum++;
			log.logln_noTimestamp("]");
//			nb.fit(list, label);

		}

		log.logln(Logger.lD,  "Loading : " + ArrayUtils.printArray(labels));
		log.logln("Data: " + dataArray);
		br.close();

		try {
			predResults = gnb.predict(dataArray);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return predResults;
	}

	// given file, load data minus label
	//return a list of results
	private static double[] getLabelsFromfile(String filePath, String fileName, 
			int fileNumber, int indexOfLabel, ArrayList<String> labelNames) throws IOException {

		String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);
		int numOfLines = FileSplitter.getLineCount(filePath,
				FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA));
	
		double[] labels = new double[numOfLines];

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;

		log.logln("Loading file " + file + ",");
		int lineNum = 0;

		//for each line, load data
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			//log.logln(Logger.lD, "Row: " + ArrayUtils.printArray(values));
			
			String label = null;

					label = values[indexOfLabel];
					labels[lineNum] = labelNames.indexOf(label);
					log.log_noTimestamp("[ label: " + label + ", index: " + labels[lineNum]);

			
			lineNum++;
			log.logln_noTimestamp("]");


		}

		br.close();

		return labels;
	}
	
	/*
	private static void getStats(String className) {
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
		log.logln(Logger.lF, "Class tested: " + className);
		log.logln("Accuracy: " + accuracy + "%");
		log.logln("Total lines training: " + gnb.getFitCount());
		log.logln("Total lines predicted: " + predictResults.length);
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
	*/


}
