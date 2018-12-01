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

import java.util.function.Function;

/**
 * Activation functions for a neural network.
 *
 * @author Javier Orbe
 */
public enum ActivationFunction {

    /**
     * Sigmoid function.
     *
     * @see <a href="https://en.wikipedia.org/wiki/Sigmoid_function" target="_top">Sigmoid function in Wikipedia</a>
     */
    SIGMOID(
            x -> 1 / (1 + Math.exp(-x)),
            y -> y * (1 - y)
    ),

    /**
     * Hyperbolic tangent.
     */
    TANH(
            Math::tanh,
            y -> 1 - (y * y)
    );

    private Function<Double, Double> function;
    private Function<Double, Double> derivative;

    /**
     * Construct an activation function.
     *
     * @param function the function.
     * @param derivative the derivative of the function.
     */
    ActivationFunction(Function<Double, Double> function, Function<Double, Double> derivative) {
        this.function = function;
        this.derivative = derivative;
    }

    /**
     * Returns the function.
     *
     * @return the function.
     */
    public Function<Double, Double> getFunction() {
        return function;
    }

    /**
     * Returns the derivative of the function.
     *
     * @return the derivative of the function.
     */
    public Function<Double, Double> getDerivative() {
        return derivative;
    }
}
