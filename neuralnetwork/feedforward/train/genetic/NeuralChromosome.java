package com.aurora.ai.neuralnetwork.feedforward.train.genetic;

import java.util.Arrays;

import com.aurora.ai.neuralnetwork.exception.NeuralNetworkError;
import com.aurora.ai.neuralnetwork.feedforward.NeuralNetwork;
import com.aurora.ai.neuralnetwork.genetic.Chromosome;
import com.aurora.ai.neuralnetwork.genetic.GeneticAlgorithm;
import com.aurora.ai.neuralnetwork.matrix.MatrixCODEC;

/**
 * NeuralChromosome: Implements a chromosome that allows a 
 * feedforward neural network to be trained using a genetic
 * algorithm.  The chromosome for a feed forward neural network
 * is the weight and threshold matrix.  
 * 
 * This class is abstract.  If you wish to train the neural
 * network using training sets, you should use the 
 * TrainingSetNeuralChromosome class.  If you wish to use 
 * a cost function to train the neural network, then
 * implement a subclass of this one that properly calculates
 * the cost.
 * 
 * The generic type GA_TYPE specifies the GeneticAlgorithm derived
 * class that implements the genetic algorithm that this class is 
 * to be used with.
 * 
 * @author Eke Stephen
 * @version 1.0
 */
abstract public class NeuralChromosome<GA_TYPE extends GeneticAlgorithm<?>>
		extends Chromosome<Double, GA_TYPE> {

	private static final Double ZERO = Double.valueOf(0);
	private static final double RANGE = 20.0;

	private NeuralNetwork network;

	/**
	 * @return the network
	 */
	public NeuralNetwork getNetwork() {
		return this.network;
	}

	public void initGenes(final int length) {
		final Double result[] = new Double[length];
		Arrays.fill(result, ZERO);
		this.setGenesDirect(result);
	}

	/**
	 * Mutate this chromosome randomly
	 */
	@Override
	public void mutate() {
		final int length = getGenes().length;
		for (int i = 0; i < length; i++) {
			double d = getGene(i);
			final double ratio = (int) ((RANGE * Math.random()) - RANGE);
			d*=ratio;
			setGene(i,d);
		}
	}



	/**
	 * Set all genes.
	 * 
	 * @param list
	 *            A list of genes.
	 * @throws NeuralNetworkException
	 */
	@Override
	public void setGenes(final Double[] list) throws NeuralNetworkError {

		// copy the new genes
		super.setGenes(list);

		calculateCost();
	}

	/**
	 * @param network
	 *            the network to set
	 */
	public void setNetwork(final NeuralNetwork network) {
		this.network = network;
	}

	public void updateGenes() throws NeuralNetworkError {
		this.setGenes(MatrixCODEC.networkToArray(this.network));
	}

	public void updateNetwork() {
		MatrixCODEC.arrayToNetwork(getGenes(), this.network);
	}

}