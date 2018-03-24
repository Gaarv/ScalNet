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

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.io.ClassPathResource
import org.scalatest.{ Matchers, WordSpec }

class ModelImportTest extends WordSpec with Matchers {

  "Importing a model saved from Keras" should {

    "return a valid DL4J model when loading Keras model" in {
      val jsonFilePath = new ClassPathResource("imdb_lstm_tf_keras_1_config.json").getFile.getAbsolutePath
      val model = NeuralNet().loadKerasModel(jsonFilePath)
      model.init()
      model.isInstanceOf[MultiLayerNetwork] shouldBe true
    }
  }

}
