package com.aurora.ai.neuralnetwork.matrix;

import com.aurora.ai.neuralnetwork.feedforward.NeuralLayer;
import com.aurora.ai.neuralnetwork.feedforward.NeuralNetwork;

/**
 * MatrixCODEC: The matrix CODEC can encode or decode a matrix
 * to/from an array of doubles.  This is very useful when the 
 * neural network must be looked at as an array of doubles for
 * genetic algorithms and simulated annealing.
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public class MatrixCODEC {

	/**
	 * Use an array to populate the memory of the neural network.
	 * 
	 * @param array
	 *            An array of doubles.
	 */
	public static void arrayToNetwork(final Double array[],
			final NeuralNetwork network) {

		// copy data to array
		int index = 0;

		for (final NeuralLayer layer : network.getLayers()) {

			// now the weight matrix(if it exists)
			if (layer.getNext() != null) {
				index = layer.getMatrix().fromPackedArray(array, index);
			}
		}
	}

	/**
	 * Convert to an array. This is used with some training algorithms that
	 * require that the "memory" of the neuron(the weight and threshold values)
	 * be expressed as a linear array.
	 * 
	 * @return The memory of the neuron.
	 */
	public static Double[] networkToArray(final NeuralNetwork network) {
		int size = 0;

		// first determine size
		for (final NeuralLayer layer : network.getLayers()) {
			// count the size of the weight matrix
			if (layer.hasMatrix()) {
				size += layer.getMatrixSize();
			}
		}

		// allocate an array to hold
		final Double result[] = new Double[size];

		// copy data to array
		int index = 0;

		for (final NeuralLayer layer : network.getLayers()) {

			// now the weight matrix(if it exists)
			if (layer.getNext() != null) {

				final Double matrix[] = layer.getMatrix().toPackedArray();
				for (int i = 0; i < matrix.length; i++) {
					result[index++] = matrix[i];
				}
			}
		}

		return result;
	}

}