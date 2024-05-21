const tf = require("@tensorflow/tfjs-node");
const InputError = require("../exceptions/InputError");

async function predictClassification(model, image) {
  try {
    // Decode dan preprocess gambar
    const tensor = tf.node.decodeJpeg(image).resizeNearestNeighbor([224, 224]).expandDims().toFloat();

    // Definisikan kelas
    const classes = ["Cancer", "Non-cancer"];

    // Prediksi gambar menggunakan model
    const prediction = model.predict(tensor);
    const score = await prediction.data(); // Ambil hasil prediksi sebagai array

    if (score.length === 0) {
      throw new Error("Model did not return any predictions.");
    }

    // Ambil nilai confidence score maksimum
    const confidenceScore = Math.max(...score) * 100;
    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    let label = classes[classResult];

    // Jika confidence score di bawah 50, kategorikan sebagai Non-cancer
    if (confidenceScore < 50) {
      label = "Non-cancer";
    }

    let explanation, suggestion;

    if (label === "Cancer") {
      suggestion = "Segera konsultasi dengan dokter terdekat.";
    } else {
      suggestion = "Anda Sehat";
    }

    return { confidenceScore, label, explanation, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

module.exports = predictClassification;
