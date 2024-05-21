const tf = require("@tensorflow/tfjs-node");

async function loadModel() {
  const modelPath = process.env.MODEL_URL;
  return tf.loadGraphModel(modelPath);
}

module.exports = loadModel;
