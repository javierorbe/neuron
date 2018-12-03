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

package com.javierorbe.neuron;

import com.javierorbe.math.Matrix;

import java.util.function.Function;

import static com.javierorbe.math.util.MathUtils.randomGaussian;

/**
 * Represents a neural network.
 *
 * @author Javier Orbe
 * @see <a href="https://en.wikipedia.org/wiki/Neural_network" target="_top">Neural network in Wikipedia</a>
 */
public class NeuralNetwork {

    private int[] layers;

    private Matrix[] weights;
    private Matrix[] biases;

    private ActivationFunction activationFunction;
    private double learningRate;

    /**
     * Construct a neural network.
     *
     * @param layers number of nodes in each layer.
     * @param activationFunction the activation function.
     * @param learningRate the learning rate.
     * @param weights the weights of the network.
     * @param biases the biases of the network.
     */
    public NeuralNetwork(int[] layers, ActivationFunction activationFunction, double learningRate, Matrix[] weights, Matrix[] biases) {
        this.layers = layers;
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;

        this.weights = weights;
        this.biases = biases;
    }
    
    /**
     * Construct a neural network with randomized values for the weights and biases.
     *
     * @param layers number of nodes in each layer.
     * @param activationFunction the activation function.
     * @param learningRate the learning rate.
     */
    public NeuralNetwork(int[] layers, ActivationFunction activationFunction, double learningRate) {
    	this.layers = layers;
    	this.activationFunction = activationFunction;
    	this.learningRate = learningRate;
        
        weights = new Matrix[layers.length - 1];
        biases = new Matrix[layers.length - 1];

        for (int i = 0; i < layers.length - 1; i++) {
            Matrix w = new Matrix(layers[i + 1], layers[i]);
            w.randomize();
            weights[i] = w;

            Matrix b = new Matrix(layers[i + 1], 1);
            b.randomize();
            biases[i] = b;
        }
    }

    /**
     * Construct a neural network with randomized values
     * with the sigmoid function as the activation function and with a learning rate of 0.1.
     *
     * @param layers number of nodes in each layer.
     */
    public NeuralNetwork(int[] layers) {
        this(layers, ActivationFunction.SIGMOID, 0.1);
    }

    /**
     * Create a getMutatedCopy of this {@code NeuralNetwork}.
     *
     * @return a getMutatedCopy of this {@code NeuralNetwork}.
     */
    public NeuralNetwork copy() {
        Matrix[] newWeights = new Matrix[weights.length];
    	Matrix[] newBiases = new Matrix[biases.length];
    	
    	for (int i = 0; i < weights.length; i++) {
    		newWeights[i] = weights[i].copy();
    	}
    	
    	for (int i = 0; i < biases.length; i++) {
    		newBiases[i] = biases[i].copy();
    	}
    	
    	return new NeuralNetwork(
    			layers.clone(), activationFunction, learningRate,
    			newWeights, newBiases);
    }
    
    /**
     * Calculate the output values of an input.
     *
     * @param input the input values.
     * @return the output values.
     */
    public double[] evaluate(double[] input) {
        Matrix result = Matrix.getColumnMatrix(input);

        for (int i = 0; i < layers.length - 1; i++) {
            result = Matrix.multiply(weights[i], result);
            result.add(biases[i]);
            result.map(this.activationFunction.getFunction());
        }

        return result.toArray();
    }

    /**
     * Train the network using backpropagation.
     *
     * @param inputArray input values.
     * @param targetArray target output values.
     * @see <a href="https://en.wikipedia.org/wiki/Backpropagation" target="_top">Backpropagation in Wikipedia</a>
     * @see <a href="https://www.youtube.com/watch?v=Ilg3gGewQ5U" target="_top">3blue1brown's video about backpropagation</a>
     */
    public void train(double[] inputArray, double[] targetArray) {
        Matrix[] layerOutput = new Matrix[layers.length];
        layerOutput[0] = Matrix.getColumnMatrix(inputArray);

        for (int i = 0; i < layers.length - 1; i++) {
            layerOutput[i + 1] = Matrix.multiply(weights[i], layerOutput[i]);
            layerOutput[i + 1].add(biases[i]);
            layerOutput[i + 1].map(activationFunction.getFunction());
        }

        Matrix target = Matrix.getColumnMatrix(targetArray);
        Matrix[] error = new Matrix[layers.length];
        error[layers.length - 1] = Matrix.subtract(target, layerOutput[layers.length - 1]);

        for (int i = layers.length - 2; i >= 0; i--) {
            Matrix gradients = Matrix.map(layerOutput[i + 1], activationFunction.getDerivative());
            gradients.multiply(error[i + 1]);
            gradients.multiply(learningRate);

            Matrix pt = layerOutput[i].transpose();
            Matrix delta = Matrix.multiply(gradients, pt);

            weights[i].add(delta);
            biases[i].add(gradients);

            Matrix t = weights[i].transpose();
            error[i] = Matrix.multiply(t, error[i + 1]);
        }
    }
    
    /**
     * Mutate the network weights and biases using the default mutation function.
     * 
     * @param rate the mutation rate.
     */
    public void mutate(double rate) {
    	final Function<Double, Double> mutation = value -> {
            if (Math.random() < rate) {
                return value + randomGaussian();
            }
            return value;
        };

    	mutate(mutation);
    }

    /**
     * Mutate the network weights and biases.
     *
     * @param mutation the function that mutates each value.
     */
    public void mutate(Function<Double, Double> mutation) {
        for (Matrix weight : weights) {
            weight.map(mutation);
        }

        for (Matrix bias : biases) {
            bias.map(mutation);
        }
    }
}
