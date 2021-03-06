package com.aurora.ai.neuralnetwork.exception;

/**
 * MatrixError: Used by the matrix classes to indicate an error.
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public class MatrixError extends RuntimeException {

	/**
	 * Serial id for this class.
	 */
	private static final long serialVersionUID = -8961386981267748942L;

	/**
	 * Construct this exception with a message.
	 * @param t The other exception.
	 */
	public MatrixError(final String message) {
		super(message);
	}

	/**
	 * Construct this exception with another exception.
	 * @param t The other exception.
	 */
	public MatrixError(final Throwable t) {
		super(t);
	}

}