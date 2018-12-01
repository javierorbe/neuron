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

import com.javierorbe.math.util.MathUtils;
import org.jboss.arquillian.container.test.api.Deployment;
import org.jboss.arquillian.junit.Arquillian;
import org.jboss.shrinkwrap.api.ShrinkWrap;
import org.jboss.shrinkwrap.api.asset.EmptyAsset;
import org.jboss.shrinkwrap.api.spec.JavaArchive;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;

@RunWith(Arquillian.class)
public class NeuralNetworkTest {

    @Deployment
    public static JavaArchive createDeployment() {
        return ShrinkWrap.create(JavaArchive.class)
                .addClass(NeuralNetwork.class)
                .addAsManifestResource(EmptyAsset.INSTANCE, "beans.xml");
    }

    @Test
    public void logicGateAND() {
        final double[][][] data = {
                {{ 0, 0 }, { 0 }},
                {{ 1, 0 }, { 0 }},
                {{ 0, 1 }, { 0 }},
                {{ 1, 1 }, { 1 }},
        };

        NeuralNetwork nn = new NeuralNetwork(new int[] {2, 5, 1});

        int randomId;
        for (int i = 0; i < 15000; i++) {
            randomId = MathUtils.randomInt(0, 3);
            nn.train(data[randomId][0], data[randomId][1]);
        }

        Assert.assertTrue(nn.evaluate(new double[]{0, 0})[0] < 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{0, 1})[0] < 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{1, 0})[0] < 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{1, 1})[0] > 0.5);
    }

    @Test
    public void logicGateOR() {
        final double[][][] data = {
                {{ 0, 0 }, { 0 }},
                {{ 1, 0 }, { 1 }},
                {{ 0, 1 }, { 1 }},
                {{ 1, 1 }, { 1 }},
        };

        NeuralNetwork nn = new NeuralNetwork(new int[] {2, 5, 1});

        int randomId;
        for (int i = 0; i < 15000; i++) {
            randomId = MathUtils.randomInt(0, 3);
            nn.train(data[randomId][0], data[randomId][1]);
        }

        Assert.assertTrue(nn.evaluate(new double[]{0, 0})[0] < 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{0, 1})[0] > 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{1, 0})[0] > 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{1, 1})[0] > 0.5);
    }

    @Test
    public void logicGateXOR() {
        final double[][][] data = {
                {{ 0, 0 }, { 0 }},
                {{ 1, 0 }, { 1 }},
                {{ 0, 1 }, { 1 }},
                {{ 1, 1 }, { 0 }},
        };

        NeuralNetwork nn = new NeuralNetwork(new int[] {2, 5, 1});

        int randomId;
        for (int i = 0; i < 15000; i++) {
            randomId = MathUtils.randomInt(0, 3);
            nn.train(data[randomId][0], data[randomId][1]);
        }

        Assert.assertTrue(nn.evaluate(new double[]{0, 0})[0] < 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{0, 1})[0] > 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{1, 0})[0] > 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{1, 1})[0] < 0.5);
    }

    @Test
    public void logicGateXNOR() {
        final double[][][] data = {
                {{ 0, 0 }, { 1 }},
                {{ 1, 0 }, { 0 }},
                {{ 0, 1 }, { 0 }},
                {{ 1, 1 }, { 1 }},
        };

        NeuralNetwork nn = new NeuralNetwork(new int[] {2, 5, 1});

        int randomId;
        for (int i = 0; i < 15000; i++) {
            randomId = MathUtils.randomInt(0, 3);
            nn.train(data[randomId][0], data[randomId][1]);
        }

        Assert.assertTrue(nn.evaluate(new double[]{0, 0})[0] > 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{0, 1})[0] < 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{1, 0})[0] < 0.5);
        Assert.assertTrue(nn.evaluate(new double[]{1, 1})[0] > 0.5);
    }
}
