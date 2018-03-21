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
