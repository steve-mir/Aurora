package com.aurora.ai.neuralnetwork.feedforward.train.genetic;

import com.aurora.ai.neuralnetwork.exception.NeuralNetworkError;
import com.aurora.ai.neuralnetwork.feedforward.NeuralNetwork;

/**
 * TrainingSetNeuralGeneticAlgorithm: Implements a genetic algorithm 
 * that allows a feedforward neural network to be trained using a 
 * genetic algorithm.  This algorithm is for a feed forward neural 
 * network.  The neural network is trained using training sets.
 * 
 * @author Eke Stephen
 * @version 1.0
 */
public class TrainingSetNeuralGeneticAlgorithm extends
		NeuralGeneticAlgorithm<TrainingSetNeuralGeneticAlgorithm> {

	protected double input[][];
	protected double ideal[][];

	public TrainingSetNeuralGeneticAlgorithm(final NeuralNetwork network,
			final boolean reset, final double input[][],
			final double ideal[][], final int populationSize,
			final double mutationPercent, final double percentToMate)
			throws NeuralNetworkError {

		this.setMutationPercent(mutationPercent);
		this.setMatingPopulation(percentToMate * 2);
		this.setPopulationSize(populationSize);
		this.setPercentToMate(percentToMate);

		this.input = input;
		this.ideal = ideal;

		setChromosomes(new TrainingSetNeuralChromosome[getPopulationSize()]);
		for (int i = 0; i < getChromosomes().length; i++) {
			final NeuralNetwork chromosomeNetwork = (NeuralNetwork) network
					.clone();
			if (reset) {
				chromosomeNetwork.reset();
			}

			final TrainingSetNeuralChromosome c = new TrainingSetNeuralChromosome(
					this, chromosomeNetwork);
			c.updateGenes();
			setChromosome(i, c);
		}
		sortChromosomes();
	}

	/**
	 * Returns the root mean square error for a complet training set.
	 * 
	 * @param len
	 *            The length of a complete training set.
	 * @return The current error for the neural network.
	 * @throws NeuralNetworkException
	 */
	public double getError() throws NeuralNetworkError {
		final NeuralNetwork network = this.getNetwork();
		return network.calculateError(this.input, this.ideal);
	}

	/**
	 * @return the ideal
	 */
	public double[][] getIdeal() {
		return this.ideal;
	}

	/**
	 * @return the input
	 */
	public double[][] getInput() {
		return this.input;
	}

}