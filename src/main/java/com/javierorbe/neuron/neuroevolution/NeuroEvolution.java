/*
 * Copyright (c) 2018 Javier Orbe
 *
 * Permission is hereby granted, free of charge, to any person obtaining a getMutatedCopy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, getMutatedCopy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.javierorbe.neuron.neuroevolution;

import java.util.ArrayList;
import java.util.List;

/**
 * Represents a neuroevolution environment.
 *
 * @param <T> The type of the elements of the population.
 *
 * @author Javier Orbe
 * @see <a href="https://en.wikipedia.org/wiki/Neuroevolution" target="_top">Neuroevolution in Wikipedia</a>
 */
public abstract class NeuroEvolution<T extends Evolvable<T>> {

    private List<T> population = new ArrayList<>();
    private int generation = 0;

    /**
     * Construct a neuroevolution environment.
     */
    public NeuroEvolution() {}

    /**
     * Returns a list with the population elements.
     *
     * @return a list with the population elements.
     */
    public List<T> getPopulation() {
        return population;
    }

    /**
     * Returns the current generation number.
     *
     * @return the current generation number.
     */
    public int getGeneration() {
        return generation;
    }

    /**
     * Continue to the next generation.
     */
    public void nextGeneration() {
        normalizeFitness();
        population = generatePopulation();
        generation++;
    }

    /**
     * Create the population for the next generation.
     *
     * @return a list of the new elements of the population.
     */
    private List<T> generatePopulation() {
        List<T> newPop = new ArrayList<>();

        for (int i = 0; i < population.size(); i++) {
            newPop.add(poolSelection());
        }

        return newPop;
    }

    /**
     * Normalize the fitness for every element of the population.
     * First, the score is reevaluated.
     * The fitness of each element is calculated dividing the score by the
     * sum of the scores of every element.
     */
    private void normalizeFitness() {
        // Score is exponentially better
        double scoreSum = population.stream().mapToDouble((elem) -> {
            double score = elem.getScore() * elem.getScore();
            elem.setScore(score);
            return score;
        }).sum();

        population.forEach((elem) -> elem.setFitness(elem.getScore() / scoreSum));
    }

    /**
     * Select a random population element, based on its fitness.
     * Elements with bigger fitness have more chances of being selected.
     *
     * @return the selected population element.
     */
    private T poolSelection() {
        int index = 0;
        double rand = Math.random();
        while (rand > 0) {
            rand -= population.get(index).getFitness();
            index += 1;
        }
        index -= 1;
        return population.get(index).getMutatedCopy();
    }
}
