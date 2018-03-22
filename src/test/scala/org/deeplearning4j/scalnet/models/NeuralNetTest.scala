/*
 * Copyright 2016 Skymind
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.deeplearning4j.scalnet.models

import org.deeplearning4j.scalnet.layers.core.Dense
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.scalatest.{ BeforeAndAfter, Matchers, WordSpec }

import scala.util.Try

/**
  * Created by maxpumperla on 19/07/17.
  */
class NeuralNetTest extends WordSpec with Matchers with BeforeAndAfter {

  var model: NeuralNet = NeuralNet()
  val shape = 100

  before {
    model = NeuralNet()
  }

  "A NeuralNet network" should {

    "produce an IllegalArgumentException when compiled without layers" in {
      intercept[java.lang.IllegalArgumentException](model.compile(null))
    }

    "not have an output layer without compiled model" in {
      model.add(Dense(shape, shape))
      Try(model.getNetwork.getOutputLayer).isFailure
    }

    "have an output layer with compiled model" in {
      model.add(Dense(shape, shape))
      model.compile(LossFunction.NEGATIVELOGLIKELIHOOD)
      Try(model.getNetwork.getOutputLayer).isSuccess
    }
  }
}
