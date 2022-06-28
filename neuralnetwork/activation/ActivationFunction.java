package com.aurora.ai.neuralnetwork.activation;

import java.io.Serializable;

/**
 * ActivationFunction: This interface allows various 
 * activation functions to be used with the feedforward
 * neural network.  Activation functions are applied
 * to the output from each layer of a neural network.
 * Activation functions scale the output into the
 * desired range. 
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public interface ActivationFunction extends Serializable {

	/**
	 * A activation function for a neural network.
	 * @param The input to the function.
	 * @return The output from the function.
	 */
	public double activationFunction(double d);

	/**
	 * Performs the derivative of the activation function function on the input.
	 * 
	 * @param d
	 *            The input.
	 * @return The output.
	 */
	public double derivativeFunction(double d);
}