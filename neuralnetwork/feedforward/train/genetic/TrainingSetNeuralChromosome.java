package com.aurora.ai.neuralnetwork.feedforward.train.genetic;

import com.aurora.ai.neuralnetwork.exception.NeuralNetworkError;
import com.aurora.ai.neuralnetwork.feedforward.NeuralNetwork;

/**
 * TrainingSetNeuralChromosome: Implements a chromosome that 
 * allows a feedforward neural network to be trained using a 
 * genetic algorithm.  The network is trained using training
 * sets.
 * 
 * The chromosome for a feed forward neural network
 * is the weight and threshold matrix.  
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public class TrainingSetNeuralChromosome extends
		NeuralChromosome<TrainingSetNeuralGeneticAlgorithm> {

	/**
	 * The constructor, takes a list of cities to set the initial "genes" to.
	 * 
	 * @param cities
	 *            The order that this chromosome would visit the cities. These
	 *            cities can be thought of as the genes of this chromosome.
	 * @throws NeuralNetworkException
	 */
	public TrainingSetNeuralChromosome(
			final TrainingSetNeuralGeneticAlgorithm genetic,
			final NeuralNetwork network) throws NeuralNetworkError {
		this.setGeneticAlgorithm(genetic);
		this.setNetwork(network);

		initGenes(network.getWeightMatrixSize());
		updateGenes();
	}

	@Override
	public void calculateCost() throws NeuralNetworkError {
		// update the network with the new gene values
		this.updateNetwork();

		// update the cost with the new genes
		final double input[][] = this.getGeneticAlgorithm().getInput();
		final double ideal[][] = this.getGeneticAlgorithm().getIdeal();

		setCost(getNetwork().calculateError(input, ideal));

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
}