package com.aurora.ai.neuralnetwork.util;


/**
 * ErrorCalculation: An implementation of root mean square (RMS)
 * error calculation.  This class is used by nearly every neural
 * network in this book to calculate error.
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public class ErrorCalculation {
	private double globalError;
	private int setSize;

	/**
	 * Returns the root mean square error for a complete training set.
	 * 
	 * @param len
	 *            The length of a complete training set.
	 * @return The current error for the neural network.
	 */
	public double calculateRMS() {
		final double err = Math.sqrt(this.globalError / (this.setSize));
		return err;

	}

	/**
	 * Reset the error accumulation to zero.
	 */
	public void reset() {
		this.globalError = 0;
		this.setSize = 0;
	}

	/**
	 * Called to update for each number that should be checked.
	 * @param actual The actual number.
	 * @param ideal The ideal number.
	 */
	public void updateError(final double actual[], final double ideal[]) {
		for (int i = 0; i < actual.length; i++) {
			final double delta = ideal[i] - actual[i];
			this.globalError += delta * delta;			
		}
		this.setSize += ideal.length;
	}

}