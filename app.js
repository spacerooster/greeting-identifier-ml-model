// more documentation available at
// https://github.com/tensorflow/tfjs-models/tree/master/speech-commands

// model files are in the same directory as index.html
const URL = "./";

async function createModel() {
    const checkpointURL = URL + "model.json";   // model topology
    const metadataURL = URL + "metadata.json";  // model metadata

    const recognizer = speechCommands.create(
        "BROWSER_FFT",  // Fourier transform type
        undefined,      // vocabulary feature (unused for TM models)
        checkpointURL,
        metadataURL
    );

    // Load model + metadata
    await recognizer.ensureModelLoaded();
    return recognizer;
}

async function init() {
    const recognizer = await createModel();
    const classLabels = recognizer.wordLabels();
    const labelContainer = document.getElementById("label-container");

    labelContainer.innerHTML = "";

    for (let i = 0; i < classLabels.length; i++) {
        labelContainer.appendChild(document.createElement("div"));
    }

    recognizer.listen(result => {
        const scores = result.scores;
        for (let i = 0; i < classLabels.length; i++) {
            const classPrediction =
                classLabels[i] + ": " + scores[i].toFixed(2);
            labelContainer.childNodes[i].innerHTML = classPrediction;
        }
    }, {
        includeSpectrogram: true,
        probabilityThreshold: 0.75,
        invokeCallbackOnNoiseAndUnknown: true,
        overlapFactor: 0.50
    });
}
