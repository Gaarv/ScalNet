package org.deeplearning4j.scalnet.layers

import org.deeplearning4j.nn.conf.{ GradientNormalization, Updater }
import org.deeplearning4j.nn.conf.distribution.Distribution
import org.deeplearning4j.nn.conf.weightnoise.WeightNoise
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.IUpdater

abstract class AbstractLSTM(forgetGateBiasInit: Double = 1.0,
                            gateActivationFn: Activation = Activation.SIGMOID,
                            weightInitRecurrent: WeightInit = WeightInit.XAVIER,
                            distRecurrent: Distribution,
                            activationFn: Activation = Activation.IDENTITY,
                            weightInit: WeightInit = WeightInit.XAVIER,
                            biasInit: Double,
                            dist: Distribution,
                            l1: Double,
                            l2: Double,
                            l1Bias: Double,
                            l2Bias: Double,
                            updater: IUpdater,
                            biasUpdater: IUpdater,
                            weightNoise: WeightNoise,
                            gradientNormalization: GradientNormalization =
                              GradientNormalization.None,
                            gradientNormalizationThreshold: Double = 1.0)
    extends BaseRecurrentLayer(
      weightInitRecurrent,
      distRecurrent,
      activationFn,
      weightInit,
      biasInit,
      dist,
      l1,
      l2,
      l1Bias,
      l2Bias,
      updater,
      biasUpdater,
      weightNoise,
      gradientNormalization,
      gradientNormalizationThreshold
    )