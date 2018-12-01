/*
 * Copyright (c) 2018 Javier Orbe
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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

import com.javierorbe.neuron.NeuralNetwork;

/**
 * Represents an element of a neuroevolution environment.
 *
 * @param <T> the type of the subclass.
 *
 * @author Javier Orbe
 */
public abstract class Evolvable<T> {

    private double score;
    private double fitness;

    private NeuralNetwork brain;

    /**
     * Construct an evolvable element.
     *
     * @param brain the neural network of the element.
     */
    public Evolvable(NeuralNetwork brain) {
        this.brain = brain;
    }

    /**
     * Returns the neural network of the element.
     *
     * @return the neural network of the element.
     */
    public NeuralNetwork getBrain() {
        return brain;
    }

    /**
     * Create a mutated copy of this element.
     *
     * @return a mutated copy of this element.
     */
    protected abstract T getMutatedCopy();

    double getScore() {
        return score;
    }

    void setScore(double score) {
        this.score = score;
    }

    /**
     * Add score to the element.
     *
     * @param score the score to add.
     */
    protected void addScore(double score) {
        this.score += score;
    }

    double getFitness() {
        return fitness;
    }

    void setFitness(double fitness) {
        this.fitness = fitness;
    }
}
