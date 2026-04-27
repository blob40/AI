public class Neuron {

    // Weights for the two inputs
    double w1, w2;

    // Bias shifts the output left/right
    double bias;


    /**
     * Initializes weights and bias randomly between -1 and 1.
     */
    public Neuron() {
        w1 = Math.random() * 2 - 1;
        w2 = Math.random() * 2 - 1;
        bias = Math.random() * 2 - 1;
    }

}
