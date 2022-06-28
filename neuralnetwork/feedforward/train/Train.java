package com.aurora.ai.neuralnetwork.feedforward.train;

import com.aurora.ai.neuralnetwork.feedforward.NeuralNetwork;

/**
 * Train: Interface for all feedforward neural network training
 * methods.  There are currently three training methods define:
 * 
 * Backpropagation
 * Genetic Algorithms
 * Simulated Annealing
 *  
 * @author Eke Stephen
 * @version 1.0
 */

public interface Train {

	/**
	 * Get the current error percent from the training.
	 * @return The current error.
	 */
	public double getError();

	/**
	 * Get the current best network from the training.
	 * @return The best network.
	 */
	public NeuralNetwork getNetwork();

	/**
	 * Perform one iteration of training.
	 */
	public void iteration();
}