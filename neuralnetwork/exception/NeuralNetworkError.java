package com.aurora.ai.neuralnetwork.exception;

/**
 * NeuralNetworkError: Used by the neural network classes to 
 * indicate an error.
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public class NeuralNetworkError extends RuntimeException {
	/**
	 * Serial id for this class.
	 */
	private static final long serialVersionUID = 7167228729133120101L;

	/**
	 * Construct a message exception.
	 * 
	 * @param msg
	 *            The exception message.
	 */
	public NeuralNetworkError(final String msg) {
		super(msg);
	}

	/**
	 * Construct an exception that holds another exception.
	 * 
	 * @param t
	 *            The other exception.
	 */
	public NeuralNetworkError(final Throwable t) {
		super(t);
	}
}