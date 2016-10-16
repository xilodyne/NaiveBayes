package xilodyne.machinelearning.classifier.examples;

import java.util.ArrayList;
import java.util.Arrays;

import mikera.arrayz.NDArray;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.machinelearning.classifier.GaussianNB;


/**
 * Gaussian NB using gender (float) and float attributes
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 */
public class GNB_Example_GenderHeight {

	private static Logger log = new Logger();

	public static void main(String[] args) {
		// G.setLoggerLevel(G.LOG_OFF);
		// G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		G.setLoggerLevel(G.LOG_DEBUG);
		log.logln_withClassName(G.lF,"");

		NDArray featuresTrain = NDArray.newArray(8, 2);
		double[] labeledDataTrain;
		GaussianNB gnb = new GaussianNB(GaussianNB.EMPTY_SAMPLES_ALLOW);

		/**
		 * CLASS
		 * male = 0.0 
		 * female = 1.0 
		 * 
		 * ATTRIBUTES
		 * Gender, height (feet), weight (lbs) 
		 * male 6 180
		 * male 5.92 (5'11") 190 
		 * male 5.58 (5'7") 170 
		 * male 5.92 (5'11") 165
		 * female 5 100 
		 * female 5.5 (5'6") 150 
		 * female 5.42 (5'5") 130 
		 * female 5.75 (5'9") 150
		 */

		/** Create NDArray of [height, weight] */
		
		featuresTrain.set(0, 0, 6.0);
		featuresTrain.set(0, 1, 180.0);

		featuresTrain.set(1, 0, 5.92);
		featuresTrain.set(1, 1, 190.0);

		featuresTrain.set(2, 0, 5.58);
		featuresTrain.set(2, 1, 170.0);

		featuresTrain.set(3, 0, 5.92);
		featuresTrain.set(3, 1, 165.0);

		featuresTrain.set(4, 0, 5.5);
		featuresTrain.set(4, 1, 100.0);

		featuresTrain.set(5, 0, 5.5);
		featuresTrain.set(5, 1, 150.0);

		featuresTrain.set(6, 0, 5.42);
		featuresTrain.set(6, 1, 130.0);

		featuresTrain.set(7, 0, 5.75);
		featuresTrain.set(7, 1, 150.0);

		/** label each correspending entry with male or female */
		labeledDataTrain = new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0 };


		try {
			gnb.fit(featuresTrain, labeledDataTrain);
		} catch (Exception e) {
			e.printStackTrace();
		}

		gnb.printAttributeValuesAndClasses();

		gnb.printMeanVar();
		System.out.println("Sample: 0, 0, 6.0, predict: " + gnb.predictSingleLabelSingleClass(0, 0, 6.0f));
		String predictedResult = gnb.predict(new ArrayList<Float>(Arrays.asList(6f, 130f)));
		System.out.println("CLASS Prediction: " + predictedResult);
	}

}
