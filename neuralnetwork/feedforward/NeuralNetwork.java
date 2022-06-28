package com.aurora.ai.neuralnetwork.feedforward;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;

import com.aurora.ai.neuralnetwork.exception.NeuralNetworkError;
import com.aurora.ai.neuralnetwork.matrix.MatrixCODEC;
import com.aurora.ai.neuralnetwork.util.ErrorCalculation;

/**
 * NeuralNetwork: This class implements a feed forward
 * neural network.  This class works in conjunction the 
 * NeuralLayer class.  Layers are added to the 
 * NeuralNetwork to specify the structure of the neural
 * network.
 * 
 * The first layer added is the input layer, the final layer added
 * is the output layer.  Any layers added between these two layers
 * are the hidden layers.
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public class NeuralNetwork implements Serializable {
	/**
	 * Serial id for this class.
	 */
	private static final long serialVersionUID = -136440631687066461L;
	
	/**
	 * The input layer.
	 */
	protected NeuralLayer inputLayer;
	
	/**
	 * The output layer.
	 */
	protected NeuralLayer outputLayer;
	
	/**
	 * All of the layers in the neural network.
	 */
	protected List<NeuralLayer> layers = new ArrayList<NeuralLayer>();
	
	/**
	 * Construct an empty neural network.
	 */
	public NeuralNetwork() {
	}

	/**
	 * Add a layer to the neural network. The first layer added is the input
	 * layer, the last layer added is the output layer.
	 * 
	 * @param layer The layer to be added.
	 */
	public void addLayer(final NeuralLayer layer) {
		// setup the forward and back pointer
		if (this.outputLayer != null) {
			layer.setPrevious(this.outputLayer);
			this.outputLayer.setNext(layer);
		}

		// update the inputLayer and outputLayer variables
		if (this.layers.size() == 0) {
			this.inputLayer = this.outputLayer = layer;
		} else {
			this.outputLayer = layer;
		}

		// add the new layer to the list
		this.layers.add(layer);
	}

	/**
	 * Calculate the error for this neural network. The error is calculated
	 * using root-mean-square(RMS).
	 * 
	 * @param input
	 *            Input patterns.
	 * @param ideal
	 *            Ideal patterns.
	 * @return The error percentage.
	 * @throws NeuralNetworkException
	 *             An error happened trying to determine the error.
	 */
	public double calculateError(final double input[][], final double ideal[][])
			throws NeuralNetworkError {
		final ErrorCalculation errorCalculation = new ErrorCalculation();

		for (int i = 0; i < ideal.length; i++) {
			computeOutputs(input[i]);
			errorCalculation.updateError(this.outputLayer.getFire(), 
					ideal[i]);
		}
		return (errorCalculation.calculateRMS());
	}

	/**
	 * Calculate the total number of neurons in the network across all layers.
	 * 
	 * @return The neuron count.
	 */
	public int calculateNeuronCount() {
		int result = 0;
		for (final NeuralLayer layer : this.layers) {
			result += layer.getNeuronCount();
		}
		return result;
	}

	/**
	 * Return a clone of this neural network. Including structure, weights and
	 * threshold values.
	 * 
	 * @return A cloned copy of the neural network.
	 */
	@Override
	public Object clone() {
		final NeuralNetwork result = cloneStructure();
		final Double copy[] = MatrixCODEC.networkToArray(this);
		MatrixCODEC.arrayToNetwork(copy, result);
		return result;
	}

	/**
	 * Return a clone of the structure of this neural network. 
	 * 
	 * @return A cloned copy of the structure of the neural network.
	 */

	public NeuralNetwork cloneStructure() {
		final NeuralNetwork result = new NeuralNetwork();

		for (final NeuralLayer layer : this.layers) {
			final NeuralLayer clonedLayer = new NeuralLayer(layer
					.getNeuronCount());
			result.addLayer(clonedLayer);
		}

		return result;
	}

	/**
	 * Compute the output for a given input to the neural network.
	 * 
	 * @param input
	 *            The input provide to the neural network.
	 * @return The results from the output neurons.
	 * @throws MatrixException A matrix error occurred.
	 * @throws NeuralNetworkException A neural network error occurred.
	 */
	public double[] computeOutputs(final double input[]) {

		if (input.length != this.inputLayer.getNeuronCount()) {
			throw new NeuralNetworkError(
					"Size mismatch: Can't compute outputs for input size="
							+ input.length + " for input layer size="
							+ this.inputLayer.getNeuronCount());
		}

		for (final NeuralLayer layer : this.layers) {
			if (layer.isInput()) {
				layer.computeOutputs(input);
			} else if (layer.isHidden()) {
				layer.computeOutputs(null);
			}
		}

		return this.outputLayer.getFire();
	}

	/**
	 * Compare the two neural networks. For them to be equal they must be of the
	 * same structure, and have the same matrix values.
	 * 
	 * @param other
	 *            The other neural network.
	 * @return True if the two networks are equal.
	 */
	public boolean equals(final NeuralNetwork other) {
		final Iterator<NeuralLayer> otherLayers = other.getLayers()
				.iterator();

		for (final NeuralLayer layer : this.getLayers()) {
			final NeuralLayer otherLayer = otherLayers.next();

			if (layer.getNeuronCount() != otherLayer.getNeuronCount()) {
				return false;
			}

			// make sure they either both have or do not have
			// a weight matrix.
			if ((layer.getMatrix() == null) && (otherLayer.getMatrix() != null)) {
				return false;
			}

			if ((layer.getMatrix() != null) && (otherLayer.getMatrix() == null)) {
				return false;
			}

			// if they both have a matrix, then compare the matrices
			if ((layer.getMatrix() != null) && (otherLayer.getMatrix() != null)) {
				if (!layer.getMatrix().equals(otherLayer.getMatrix())) {
					return false;
				}
			}
		}

		return true;
	}

	/**
	 * Get the count for how many hidden layers are present.
	 * @return The hidden layer count.
	 */
	public int getHiddenLayerCount() {
		return this.layers.size() - 2;
	}

	/**
	 * Get a collection of the hidden layers in the network.
	 * @return The hidden layers.
	 */
	public Collection<NeuralLayer> getHiddenLayers() {
		final Collection<NeuralLayer> result = new ArrayList<NeuralLayer>();
		for (final NeuralLayer layer : this.layers) {
			if (layer.isHidden()) {
				result.add(layer);
			}
		}
		return result;
	}

	/**
	 * Get the input layer.
	 * @return The input layer.
	 */
	public NeuralLayer getInputLayer() {
		return this.inputLayer;
	}

	/**
	 * Get all layers.
	 * @return All layers.
	 */
	public List<NeuralLayer> getLayers() {
		return this.layers;
	}

	/**
	 * Get the output layer.
	 * @return The output layer.
	 */
	public NeuralLayer getOutputLayer() {
		return this.outputLayer;
	}

	/**
	 * Get the size of the weight and threshold matrix.
	 * @return The size of the matrix.
	 */
	public int getWeightMatrixSize() {
		int result = 0;
		for (final NeuralLayer layer : this.layers) {
			result += layer.getMatrixSize();
		}
		return result;
	}

	/**
	 * Reset the weight matrix and the thresholds.
	 * 
	 * @throws MatrixException
	 */
	public void reset() {
		for (final NeuralLayer layer : this.layers) {
			layer.reset();
		}
	}
}