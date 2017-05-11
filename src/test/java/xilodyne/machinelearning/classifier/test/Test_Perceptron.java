package xilodyne.machinelearning.classifier.test;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import xilodyne.machinelearning.classifier.neural.Perceptron;
import xilodyne.machinelearning.classifier.neural.Perceptron_Int;
import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
//import xilodyne.util.Logger;

public class Test_Perceptron {
	
	//private Logger log = new Logger();


	public static void main(String[] args) {
		// G.setLoggerLevel(G.LOG_OFF);
		G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);

		
		// create a perceptron of 2 w, 2 i, 0 t
		Perceptron ptron1, ptron2, ptron3, ptron4 = null;

		float fThreshold = 0;
		float[] fWeights = new float[] { 1f, 1f, 1f };
		ptron1 = new Perceptron(fWeights, fThreshold);
		
		//float[] fInputs = new float[] { 1f, -1f };
		ptron1.fit(new float[] {2f, 0, -3f}, 1f);
		//System.out.println("Given input: " + fInputs[0] + "," + fInputs[1] + " activated: " + ptron.predict(fInputs));

		System.out.println("\n\n");
		fWeights = new float[] { 1f, 2f, 3f };
		ptron2 = new Perceptron(fWeights, fThreshold);
		ptron2.fit(new float[] {3f, 2f, 1f}, 0);		
		ptron2.fit(new float[] {4, 0, -1f}, 0);

		System.out.println("\n\n");
		fWeights = new float[] { 3f, 0, 2f };
		ptron3 = new Perceptron(fWeights, fThreshold);
		ptron3.fit(new float[] {2f, -2f, 4f}, 0);		
		ptron3.fit(new float[] {-1f, -3f, 2f}, 1);		
		ptron3.fit(new float[] {0f, 2f, 1f}, 0);		
		ptron3.fit(new float[] {2f, -2f, 4f}, 0);		
		ptron3.fit(new float[] {-1f, -3f, 2f}, 1);		
		ptron3.fit(new float[] {2f, -2f, 4f}, 0);		

		System.out.println("\n\n");
		fWeights = new float[] { 3.2f, 3.2f};
		ptron4 = new Perceptron(fWeights, fThreshold);
		ptron4.fit(new float[] {2f, 2f}, 4);		
		ptron4.fit(new float[] {2f, 2f}, 4);		

	}
	
	@Test
	public void check_PerceptonTrueFalse() {
		// G.setLoggerLevel(G.LOG_OFF);
		G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);

		System.out.println();
		System.out.println("*** TEST *** Check Perceptron True/False");
		System.out.println();
		
		Perceptron_Int ptronInt = null;

		int threshold = 0;
		int[] weights = new int[] { 1, 2 };
		ptronInt = new Perceptron_Int(weights, threshold);
		
		int[] inputs = new int[] { 1, -1 };
		int result = ptronInt.activate(inputs) ? 1:0;		
		assertEquals(0, result, 0);
//		System.out.println("Given input: " + inputs[0] + "," + inputs[1] + " activated: " + ptron.activate(inputs));
		
		inputs = new int[] {-1, 1};
		result = ptronInt.activate(inputs) ? 1:0;		
		assertEquals(1, result, 0);
		//System.out.println("Given input: " + inputs[0] + "," + inputs[1] + " activated: " + ptron.activate(inputs));
		
		inputs = new int[] {2, -1};
		result = ptronInt.activate(inputs) ? 1:0;		
		assertEquals(0, result, 0);
		//System.out.println("Given input: " + inputs[0] + "," + inputs[1] + " activated: " + ptron.activate(inputs));


		
	}
	
	@Test
	public void check_PerceptonUpdateWeights() {
		// G.setLoggerLevel(G.LOG_OFF);
		G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);

		System.out.println();
		System.out.println("*** TEST *** Check Perceptron update weights");
		System.out.println();
		
		Perceptron ptron = null;


		float fThreshold = 0;
		float[] fWeights = new float[] { 1f, 1f, 1f };
		ptron = new Perceptron(fWeights, fThreshold);
		float [] values = new float[]{2f, 0, -3f};
		
		System.out.println("predict: " + ArrayUtils.printArray(values));
		assertEquals(0,  ptron.predict(values),  0);

		ptron.fit(values, 1f);
		System.out.println("predict: " + ArrayUtils.printArray(values));
		assertEquals(1.0,  ptron.predict(values),  0);
//		System.out.println("Update activated: " + ptron.activate(inputs));

	}


	public static void mainNew(String[] args) {
		// G.setLoggerLevel(G.LOG_OFF);
		G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);

		// create a perceptron of 2 w, 2 i, 0 t

		Perceptron_Int ptron = null;

		int threshold = 0; //bias
		int[] weights = new int[] { 1, 2 };  //weights match input size
		ptron = new Perceptron_Int(weights, threshold);

		//List <Integer> data = new ArrayList<Integer>(Arrays.asList(-1, 1));
		
		int[] inputs = new int[] { 1, -1 };
		System.out.println("Given input: " + inputs[0] + "," + inputs[1] + " activated: " + ptron.activate(inputs));
		
		assertEquals("t", ptron, 0);
		System.out.println("*** TEST COMPLETE ***");

		inputs = new int[] {-1, 1};
		System.out.println("Given input: " + inputs[0] + "," + inputs[1] + " activated: " + ptron.activate(inputs));
		
		inputs = new int[] {2, -1};
		System.out.println("Given input: " + inputs[0] + "," + inputs[1] + " activated: " + ptron.activate(inputs));
	
	}

}
