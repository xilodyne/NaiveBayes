package xilodyne.machinelearning.classifier.neural;

public class Perceptron_Int {
	
	private int[] weights = null;
	private int threshold = 0;
	
	//assign weights and threshold
	public Perceptron_Int(int[] newWeights, int newThreshold) {
		this.weights = newWeights;
		this.threshold = newThreshold;
	}
	
	//give inputs, matching the number of weights
	//return true if activated
	public boolean activate(int[] inputs) {
		boolean activated = false;
		
		int dotP = this.dotProduct(inputs, this.weights);
		
		if (dotP > this.threshold) {
			activated = true;
		}
		
		return activated;
	}
	
	
	
	
	/*
	    x1 = inputs[0]
        x2 = inputs[1]
        w1 = self.weights[0]
        w2 = self.weights[1]
        p1 = x1 * w1
        p2 = x2 * w2
        p = p1 + p2
	 */
	private int dotProduct(int[] arrayOne, int[] arrayTwo) {
		//assuming arrays are same length
		int product = 0;
		
		for (int x = 0; x < arrayOne.length; x++ ) {
			product = product + ( arrayOne[x] * arrayTwo[x]);
		}
		
		return product;
		
	}
	
	public void fit(int[] inputList, int[] labels) {
		//given list of inputs and labels, 
		//get perceptron answer, compare to label, 
		//if answer vs label is different, perform weight update
		
		
	}
	
	
	

}
