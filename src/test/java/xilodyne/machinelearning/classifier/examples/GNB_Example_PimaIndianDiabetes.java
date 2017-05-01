package xilodyne.machinelearning.classifier.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import mikera.arrayz.NDArray;
import xilodyne.util.io.FileSplitter;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.machinelearning.classifier.bayes.GaussianNB;

/**
 * Gaussian NB using Pima Indian Diabetes Data Set.
 * https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
 * 
 * Uses NDArray by vectorz https://github.com/mikera/vectorz
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.2
 * 
 */
public class GNB_Example_PimaIndianDiabetes {

	private static Logger log = new Logger();


	// get metadata,
	// open file,
	// get number of columns
	// get number of rows
	// assume last column are class
	static NDArray trainingData = null;
	static double[] trainingLabels = null;

	public static void main(String[] args) {
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		 G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);
		log.logln_withClassName(G.lF,"");

		String filePath = "./test-data";
		String fileName = "pima-indians-diabetes.csv";
		try {
			GaussianNB gnb = new GaussianNB(GaussianNB.EMPTY_SAMPLES_IGNORE);

			FileSplitter.createSubFiles(5, filePath, fileName, FileSplitter.fileExtCSV);		
			load2D_NDArray(filePath, fileName, 5, 9);
			gnb.fit(trainingData, trainingLabels);

			log.logln_withClassName(G.lD, "Output of ND Array...");
			log.logln("\ndata: "+ trainingData);
			log.logln("ND array dim: " + trainingData.getShape(1));
			
			// printlabeledData();


			load2D_NDArray(filePath, fileName, 4, 9);
			gnb.fit(trainingData, trainingLabels);

			log.logln(G.lD, "Output of ND Array...");
			log.logln(trainingData.toString());
			log.logln("ND array size: " + trainingData.getShape(1));
			
			// printlabeledData();
			load2D_NDArray(filePath, fileName, 3, 9);
			gnb.fit(trainingData, trainingLabels);
			load2D_NDArray(filePath, fileName, 2, 9);
			gnb.fit(trainingData, trainingLabels);

			load2D_NDArray(filePath, fileName, 1, 9);

			double[] predictedResults = gnb.predict(trainingData);
			log.logln("Predicted Results size: " + predictedResults.length);
			log.logln("Class Labels size: " + trainingLabels.length);
			

			double accuracy = gnb.getAccuracyOfPredictedResults(trainingLabels, predictedResults);
			System.out.println("Accuracy: " + accuracy);

		} catch (IOException e) {
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	// load the data into the arrays
	// assuming the last value of the line is the labeled Class value
	private static void load2D_NDArray(String filePath, String fileName, int fileNumber, int numOfFeatures) throws IOException {

		String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtCSV);
		int numOfLines = FileSplitter.getLineCount(filePath,
				FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtCSV));
		
		trainingData = NDArray.newArray(numOfLines, (numOfFeatures - 1));
		trainingLabels = new double[numOfLines];

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;
		int lineNum = 0;

		log.logln("Loading file " + file + ",");
		log.logln("size is " + numOfLines);

		while ((line = br.readLine()) != null) {
			values = line.split(",");
			// load the last value into the class array
			for (int valueIndex = 0; valueIndex < numOfFeatures; valueIndex++) {
				if (valueIndex == (numOfFeatures - 1)) {
					trainingLabels[lineNum] = Float.valueOf(values[valueIndex]);
					log.logln(valueIndex + ":" + values[valueIndex]);
				} else {
					trainingData.set(lineNum, valueIndex, Double.valueOf(values[valueIndex]));
					log.logln(valueIndex + ":" + values[valueIndex] + ", \t");
				}
			}
			lineNum++;
		}
		br.close();
	}
}
