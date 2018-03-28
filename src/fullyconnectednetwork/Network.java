package fullyconnectednetwork;

import parser.Attribute;
import parser.Node;
import parser.Parser;
import parser.ParserTools;
import sun.nio.ch.Net;
import trainset.TrainSet;

import java.io.*;
import java.util.Arrays;

/**
 * Created by Owner on 3/27/2018.
 */
public class Network {
    public double[][] outputs;
    public double[][][] weights;
    public double[][] bias;
    public double[][] error_signal;
    public double[][] output_derivative;


    public final int[] NETWORK_LAYER_SIZES;
    public final int INPUT_SIZE;
    public final int OUT_SIZE;
    public final int NETWORK_SIZE;

    public Network(int... NETWORK_LAYER_SIZES) {
        this.NETWORK_LAYER_SIZES = NETWORK_LAYER_SIZES;
        this.INPUT_SIZE = NETWORK_LAYER_SIZES[0];
        this.NETWORK_SIZE = NETWORK_LAYER_SIZES.length;
        this.OUT_SIZE = NETWORK_LAYER_SIZES[NETWORK_SIZE -1];


        this.outputs = new double[NETWORK_SIZE][];
        this.weights = new double[NETWORK_SIZE][][];
        this.bias = new double[NETWORK_SIZE][];
        this.error_signal = new double[NETWORK_SIZE][];
        this.output_derivative = new double[NETWORK_SIZE][];


        for(int i = 0; i < NETWORK_SIZE; i++){
            this.outputs[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.error_signal[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.output_derivative[i] = new double[NETWORK_LAYER_SIZES[i]];
            this.bias[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], -0.5,0.7);


            if(i > 0){
                weights[i] = NetworkTools.createRandomArray(NETWORK_LAYER_SIZES[i], NETWORK_LAYER_SIZES[i -1], -1, 1);
            }
        }
    }

    public double[] calculate(double... input){
        if(input.length != this.INPUT_SIZE){
            return null;
        }

        this.outputs[0] = input;

        for(int layer = 1; layer < NETWORK_SIZE; layer++){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){

                double sum = bias[layer][neuron];
                for(int prevousNeuron = 0; prevousNeuron < NETWORK_LAYER_SIZES[layer-1]; prevousNeuron++){
                    sum += outputs[layer-1][prevousNeuron] *weights[layer][neuron][prevousNeuron];
                }

                outputs[layer][neuron] = sigmoid(sum);
                output_derivative[layer][neuron] = outputs[layer][neuron] * (1 - outputs[layer][neuron]);
            }
        }
        return outputs[NETWORK_SIZE -1];
    }

    public void train(TrainSet set, int loops, int batch_size){
        if(set.INPUT_SIZE != INPUT_SIZE || set.OUTPUT_SIZE != OUT_SIZE) return;
        for(int i = 0; i < loops; i++){
            TrainSet batch = set.extractBatch(batch_size);
            for(int b = 0; b<batch.size(); b++){
                this.train(batch.getInput(b), batch.getOutput(b), 0.3);
            }
            System.out.println(MSE(batch));
        }
    }

    public void train(double[] input, double[] target, double eta){
        if(input.length != INPUT_SIZE || target.length != OUT_SIZE)return;
        calculate(input);
        backpropError(target);
        updateWeights(eta);
    }

    public double MSE(double[] input, double[] target){
        if(input.length != INPUT_SIZE || target.length != OUT_SIZE) return 0;
        calculate(input);
        double v = 0;
        for(int i = 0; i < target.length; i++){
            v += (target[i] - outputs[NETWORK_SIZE -1][i]) * (target[i] - outputs[NETWORK_SIZE -1][i]);
        }
        return v / 2d * (target.length);
    }

    public double MSE(TrainSet set){
        double v = 0;
        for(int i = 0; i < set.size(); i++){
            v += MSE(set.getInput(i), set.getOutput(i));
        }
        return v / set.size();
    }

    public void backpropError(double[] target){
        for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[NETWORK_SIZE -1]; neuron++){
            error_signal[NETWORK_SIZE -1][neuron] = (outputs[NETWORK_SIZE -1][neuron] - target[neuron])
                    * output_derivative[NETWORK_SIZE -1][neuron];
        }

        for (int layer = NETWORK_SIZE-2; layer > 0; layer--){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){
                double sum = 0;
                for(int nextNeuron = 0; nextNeuron < NETWORK_LAYER_SIZES[layer+1]; nextNeuron++){
                    sum += weights[layer +1][nextNeuron][neuron] * error_signal[layer +1] [nextNeuron];
                }
                this.error_signal[layer][neuron] = sum * output_derivative[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta){
        for(int layer = 1; layer < NETWORK_SIZE; layer++){
            for(int neuron = 0; neuron < NETWORK_LAYER_SIZES[layer]; neuron++){
                double delta = -eta * error_signal[layer][neuron];
                bias[layer][neuron] += delta;

                for(int prevNeuron = 0; prevNeuron < NETWORK_LAYER_SIZES[layer -1]; prevNeuron++){
                    weights[layer][neuron][prevNeuron] += delta * outputs[layer -1][prevNeuron];
                }
            }
        }
    }

    private double sigmoid(double x){
        return 1/ (1 + Math.exp(-x));

    }

//    public static void mainE(String[] args){
//        Network net = new Network(4, 3, 3, 2);
//        int epoch = 10000;
//
//        double[] in = new double[]{0.1, 0.2, 0.3, 0.4 };
//        double[] target = new double[]{0.9, 0.1};
//
//        double[] in2 = new double[]{0.6, 0.1, 0.4, 0.8};
//        double[] target2 = new double[]{0.1, 0.9};
//
//        for(int i = 0; i <epoch; i++){
//            net.train(in, target, 0.3);
//            net.train(in2, target2, 0.3);
//        }
//
//        System.out.println(Arrays.toString(net.calculate(in)));
//        System.out.println(Arrays.toString(net.calculate(in2)));
//    }

//    public static void mainF(String[] args){
//
//        /*
//          Batch size only useful with a bunch of data
//         */
//        Network net = new Network(4, 3, 3, 2);
//        int epoch = 100000;
//
//        TrainSet set = new TrainSet(4,2);
//        set.addData(new double[]{0.1, 0.2, 0.3, 0.4}, new double[]{0.9, 0.1});
//        set.addData(new double[]{0.9, 0.8, 0.7, 0.6}, new double[]{0.1, 0.9});
//        set.addData(new double[]{0.3, 0.8, 0.1, 0.4}, new double[]{0.3, 0.7});
//        set.addData(new double[]{0.9, 0.8, 0.1, 0.2}, new double[]{0.7, 0.3});
//
//        net.train(set, epoch, 4);
//
//        for(int i = 0; i < 4; i++){
//            System.out.println(Arrays.toString(net.calculate(set.getInput(i))));
//        }
//
//    }

    public static void main(String[] args){
        try {
//            Network net = new Network(4,3,2);
//            net.saveNetwork("res/test2.txt");
            Network net = Network.loadNetwork("res/test2.txt");
        }catch(Exception e){
            e.printStackTrace();
        }
    }


    public void saveNetwork(String fileName) throws Exception {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("sizes", Arrays.toString(this.NETWORK_LAYER_SIZES)));
        netw.addChild(ly);
        root.addChild(netw);
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {

            Node c = new Node("" + layer);
            ly.addChild(c);
            Node w = new Node("weights");
            Node b = new Node("biases");
            c.addChild(w);
            c.addChild(b);

            b.addAttribute("values", Arrays.toString(this.bias[layer]));

            for (int we = 0; we < this.weights[layer].length; we++) {

                w.addAttribute("" + we, Arrays.toString(weights[layer][we]));
            }
        }
        p.close();
    }

    public static Network loadNetwork(String fileName) throws Exception {

        Parser p = new Parser();

        p.load(fileName);
        String sizes = p.getValue(new String[] { "Network" }, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        Network ne = new Network(si);

        for (int i = 1; i < ne.NETWORK_SIZE; i++) {
            String biases = p.getValue(new String[] { "Network", "Layers", new String(i + ""), "biases" }, "values");
            double[] bias = ParserTools.parseDoubleArray(biases);
            ne.bias[i] = bias;

            for(int n = 0; n < ne.NETWORK_LAYER_SIZES[i]; n++){

                String current = p.getValue(new String[] { "Network", "Layers", new String(i + ""), "weights" }, ""+n);
                double[] val = ParserTools.parseDoubleArray(current);

                ne.weights[i][n] = val;
            }
        }
        p.close();
        return ne;

    }
}
