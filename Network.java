import java.util.ArrayList;
import java.util.Random;

class Network{
    ArrayList<double[][]> weights = new ArrayList<double[][]>();
    ArrayList<double[][]> dwT = new ArrayList<double[][]>();
    ArrayList<double[]> biases = new ArrayList<double[]>();
    ArrayList<double[]> dbT = new ArrayList<double[]>();
    ArrayList<double[]> activations = new ArrayList<double[]>();
    ArrayList<double[]> zVal = new ArrayList<double[]>();
    ArrayList<Integer> Layers = new ArrayList<Integer>();
    ArrayList<Double> lowCosts = new ArrayList<Double>();
    int count = 0;
    int numLayers = 3;
    double c = 10000;

    public Network(){
        Random rand = new Random();
        Layers.add(784);
        Layers.add(100);
        //Layers.add();
        //Layers.add(400);
        Layers.add(10);
        double[][] w = new double[0][0];
        for(int i = 0; i <  numLayers - 1; i++){
            w = new double[Layers.get(i + 1)][Layers.get(i)];
            for(int j = 0; j < Layers.get(i + 1); j++){
                for(int k = 0; k < Layers.get(i); k++){

                    w[j][k] = rand.nextGaussian(0, 1 / Math.pow(784, 0.5));
                }
            }
            weights.add(w);
        }

        double[][] dwi = new double[0][0];
        for(int i = 0; i <  numLayers - 1; i++){
            dwi = new double[Layers.get(i + 1)][Layers.get(i)];
            for(int j = 0; j < Layers.get(i + 1); j++){
                for(int k = 0; k < Layers.get(i); k++){
                    dwi[j][k] = 0.0;
                }
            }
            dwT.add(dwi);
        }

        double[] b = new double[0];
        for(int i = 0; i < numLayers - 1; i++){
            b = new double[Layers.get(i + 1)];
            for(int j = 0; j < Layers.get(i + 1); j++){
                b[j] = rand.nextGaussian();
            }
            biases.add(b);
        }

        double[] dbi = new double[0];
        for(int i = 0; i < numLayers - 1; i++){
            dbi = new double[Layers.get(i + 1)];
            for(int j = 0; j < Layers.get(i + 1); j++){
                dbi[j] = 0.0;
            }
            dbT.add(dbi);
        }

        double[] a = new double[0];
        for(int i = 0; i < Layers.size(); i++){
            a = new double[Layers.get(i)];
            for(int j = 0; j < Layers.get(i); j++){
                if(i == 0){
                    a[j] = rand.nextGaussian();
                }
                else{
                    a[j] = 0.0;
                }
            }
            activations.add(a);
        }
        double[] z = new double[0];
        for(int i = 0; i < Layers.size(); i++){
            z = new double[Layers.get(i)];
            for(int j = 0; j < Layers.get(i); j++){
                z[j] = 0.0;
            }
            zVal.add(z);
        }
    }

    public double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public double sigmoidDerivative(double x){
        return Math.exp(-x) / Math.pow(1 + Math.exp(-x), 2);
    }

    public void costB(double x){
        c += -1 * Math.log(x);
    }

    public void cost(double x){
        c += -1 * Math.log(1 - x);
    }

    public double costDerivativeB(double x){
        return -1 / x;
    }

    public double costDerivative(double x){
        return 1 / (1 - x);
    }

    public double getCost(){
        return c;
    }

    public void setCost(double cSub){
        c = cSub;
    }

    public void feedForward(double[] imageData, double label){
        Random rand = new Random();
        for(int i = 0; i < Layers.size(); i++){
            for(int j = 0; j < Layers.get(i); j++){
                if(i == 0){
                    activations.get(i)[j] = sigmoid(imageData[j] / 100);
                    // sigmoid(imageData[j] / 100)
                }
                else{
                    activations.get(i)[j] = 0.0;
                }
            }
        }
        for(int i = 0; i < Layers.size(); i++){
            for(int j = 0; j < Layers.get(i); j++){
                zVal.get(i)[j] = 0.0;
            }
        }


        for(int i = 1; i < Layers.size(); i++){
            for(int j = 0; j < activations.get(i).length; j++){
                //zVal.get(i)[j] = 0.0;
                for(int k = 0; k < activations.get(i - 1).length; k++){
                    //System.out.println(zVal.size());
                    zVal.get(i)[j] += activations.get(i - 1)[k] * weights.get(i - 1)[j][k];
                    //System.out.println("Hello");
                }
                zVal.get(i)[j] += biases.get(i - 1)[j];
                activations.get(i)[j] = sigmoid(zVal.get(i)[j]);
            }
        }
        for(int i = 0; i < activations.get(Layers.size() - 1).length; i++){
            if(((int) label) == i){
                costB(activations.get(Layers.size() - 1)[i]);
            }
            else {
                cost(activations.get(Layers.size() - 1)[i]);
            }

        }

    }

    public void backPropogation(double label){
        ArrayList<double[][]> dw = new ArrayList<double[][]>();
        ArrayList<double[]> db = new ArrayList<double[]>();

        double[][] dwi = new double[weights.get(Layers.size() - 2).length][weights.get(Layers.size() - 2)[0].length];
        for(int i = 0 ; i < weights.get(Layers.size() - 2).length; i++){
            for(int j = 0; j < weights.get(Layers.size() - 2)[0].length; j++){
                if(i == ((int) label)){
                    dwi[i][j] = costDerivativeB(activations.get(Layers.size() - 1)[i]) * sigmoidDerivative(zVal.get(Layers.size() - 1)[i]) * activations.get(Layers.size() - 2)[j];
                    //costDerivativeB(activations.get(Layers.size() - 1)[i]) * sigmoidDerivative(zVal.get(Layers.size() - 1)[i]) * activations.get(Layers.size() - 2)[j];
                }
                else {
                    dwi[i][j] = costDerivative(activations.get(Layers.size() - 1)[i]) * sigmoidDerivative(zVal.get(Layers.size() - 1)[i]) * activations.get(Layers.size() - 2)[j];
                    //costDerivative(activations.get(Layers.size() - 1)[i]) * sigmoidDerivative(zVal.get(Layers.size() - 1)[i]) * activations.get(Layers.size() - 2)[j];
                }
            }
        }
        dw.add(dwi);
        for(int i = Layers.size() - 3; i > -1; i--){
            dwi = new double[weights.get(i).length][weights.get(i)[0].length];
            for(int j = 0; j < weights.get(i).length; j++){
                for(int k = 0; k < weights.get(i)[0].length; k++){
                    dwi[j][k] = 0.0;
                    for(int l = 0; l < weights.get(i + 1).length; l++){
                        //System.out.println("hello");
                        dwi[j][k] += dw.get(0)[l][j] * weights.get(i + 1)[l][j] / activations.get(i + 1)[j] * sigmoidDerivative(zVal.get(i + 1)[j]) * activations.get(i)[k];
                    }
                }
            }
            dw.add(0, dwi);
        }
        double[] dbi = new double[biases.get(Layers.size() - 2).length];
        for(int i = 0; i < biases.get(Layers.size() - 2).length; i++){
            if(i == ((int) label)){
                dbi[i] = costDerivativeB(activations.get(Layers.size() - 1)[i]) * sigmoidDerivative(zVal.get(Layers.size() - 1)[i]);
                // costDerivativeB(activations.get(Layers.size() - 1)[i]) * sigmoidDerivative(zVal.get(Layers.size() - 1)[i]);
            }
            else {
                dbi[i] = costDerivative(activations.get(Layers.size() - 1)[i]) * sigmoidDerivative(zVal.get(Layers.size() - 1)[i]);
                // costDerivative(activations.get(Layers.size() - 1)[i]) * sigmoidDerivative(zVal.get(Layers.size() - 1)[i]);
            }
        }
        db.add(dbi);
        for(int i = Layers.size() - 3; i > -1; i--){
            dbi = new double[biases.get(i).length];
            for(int j = 0; j < biases.get(i).length; j++){
                dbi[j] = 0.0;
                for(int k = 0; k < biases.get(i + 1).length; k++){
                    dbi[j] += db.get(0)[k] * weights.get(i + 1)[k][j] * sigmoidDerivative(zVal.get(i + 1)[j]);
                   // / activations.get(i + 1)[j]
                }
            }
            db.add(0, dbi);
        }
        for(int i = 0; i < Layers.size() - 1; i++){
            for(int j = 0; j < weights.get(i).length; j++){
                for(int k = 0; k < weights.get(i)[0].length; k++){
                    dwT.get(i)[j][k] += dw.get(i)[j][k];
                }
            }
        }
        //System.out.println(dwT.get(0).length);
        //System.out.println(dwT.get(1).length);
        //System.out.println(dwT.get(2).length);
        for(int i = 0; i < Layers.size() - 1; i++){
            for(int j = 0; j < biases.get(i).length; j++){
                dbT.get(i)[j] += db.get(i)[j];
            }
        }
    }

    public void subtract(double learningRate){
        c /= 100;
        for(int i = 0; i < Layers.size() - 1; i++){
            for(int j = 0; j < weights.get(i).length; j++){
                for(int k = 0; k < weights.get(i)[0].length; k++){
                    weights.get(i)[j][k] -= (learningRate * (dwT.get(i)[j][k] / 100));
                }
            }
        }
        for(int i = 0; i < Layers.size() - 1; i++){
            for(int j = 0; j < biases.get(i).length; j++){
                biases.get(i)[j] -= (learningRate * (dbT.get(i)[j] / 100));
            }
        }
        dwT.clear();
        dbT.clear();
        double[][] dwi = new double[0][0];
        for(int i = 0; i <  numLayers - 1; i++){
            dwi = new double[Layers.get(i + 1)][Layers.get(i)];
            for(int j = 0; j < Layers.get(i + 1); j++){
                for(int k = 0; k < Layers.get(i); k++){
                    dwi[j][k] = 0.0;
                }
            }
            dwT.add(dwi);
        }
        double[] dbi = new double[0];
        for(int i = 0; i < numLayers - 1; i++){
            dbi = new double[Layers.get(i + 1)];
            for(int j = 0; j < Layers.get(i + 1); j++){
                dbi[j] = 0.0;
            }
            dbT.add(dbi);
        }
    }

    public void AnZ(double label){
        System.out.println("count: " + count);
        System.out.println("Label: " + label);
        for(int i = 0; i < activations.size(); i++){
            System.out.print("Activation Layer " + i + ": ");
            for(int j = 0; j < activations.get(i).length; j++){
                System.out.print(activations.get(i)[j] + ", ");
            }
            System.out.println();
        }
        for(int i = 0; i < zVal.size(); i++){
            System.out.print("zVal Layer " + i + ": ");
            for(int j = 0; j < zVal.get(i).length; j++){
                System.out.print(zVal.get(i)[j] + ", ");
            }
            System.out.println();
        }
        count++;
    }

    public boolean compare(double label){
        double greatest = 0.0;
        double preLabel = 0.0;
        for(int i = 0; i < activations.get(2).length; i++){
            if(greatest <= activations.get(2)[i]){
                greatest = activations.get(2)[i];
                preLabel = i;
            }
        }
        if(preLabel == label){
            return true;
        }
        else{
            return false;
        }
    }

    public void addCost(double addition){
        lowCosts.add(addition);
    }

    public int getSize(){
        return lowCosts.size();
    }

    public String toString(){
        String s = "";
        for(int i = 0 ; i < weights.size(); i++){
            for(int j = 0; j < weights.get(i).length; j++){
                for(int k = 0; k < weights.get(i)[j].length; k++){
                    s += weights.get(i)[j][k] + " ";
                }
                s += "\n";
            }
            s += "\n";
        }
        s += "\n";
        s += "\n";
        for(int i = 0; i < biases.size(); i++){
            for(int j = 0; j < biases.get(i).length; j++){
                s += biases.get(i)[j] + " ";
            }
            s += "\n";
        }
        return s;
    }

    public String toString2(){
        String s = "";
        for(int i = 0 ; i < dwT.size(); i++){
            for(int j = 0; j < dwT.get(i).length; j++){
                for(int k = 0; k < dwT.get(i)[j].length; k++){
                    s += dwT.get(i)[j][k] + " ";
                }
                s += "\n";
            }
            s += "\n";
        }
        s += "\n";
        s += "\n";
        for(int i = 0; i < dbT.size(); i++){
            for(int j = 0; j < dbT.get(i).length; j++){
                s += dbT.get(i)[j] + " ";
            }
            s += "\n";
        }
        return s;
    }

}


