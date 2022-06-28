package com.aurora.ai.neuralnetwork.activation;

import com.aurora.ai.neuralnetwork.util.BoundNumbers;

/**
 * ActivationSigmoid: The sigmoid activation function takes on a
 * sigmoidal shape.  Only positive numbers are generated.  Do not
 * use this activation function if negative number output is desired.
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public class ActivationSigmoid implements ActivationFunction {
	/**
	 * Serial id for this class.
	 */
	private static final long serialVersionUID = 5622349801036468572L;

	/**
	 * A threshold function for a neural network.
	 * @param The input to the function.
	 * @return The output from the function.
	 */
	public double activationFunction(final double d) {
		return 1.0 / (1 + BoundNumbers.exp(-1.0 * d));
	}
	
	/**
	 * Some training methods require the derivative.
	 * @param The input.
	 * @return The output.
	 */
	public double derivativeFunction(double d) {
		return d*(1.0-d);
	}

}