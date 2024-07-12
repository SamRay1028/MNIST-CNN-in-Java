import java.util.ArrayList;
import java.util.*;

public class Main {
    public static void main(String[] args) {
        //System.out.println("Hello world!");
        //System.out.println("Hello world!");
        Network net = new Network();
        ArrayList<Image> trainImages = new DataReader().readData("/Users/samraya/IdeaProjects/Neural Network 3.1/Data/mnist_train.csv");
        ArrayList<Image> testImages = new DataReader().readData("/Users/samraya/IdeaProjects/Neural Network 3.1/Data/mnist_test.csv");
        ArrayList<ArrayList<Image>> trainImageSets = new ArrayList<ArrayList<Image>>();
        ArrayList<Image> validationImages = new ArrayList<Image>();
        ArrayList<Image> holder = new ArrayList<Image>();
        double[] costs = new double[10];
        double[] previousCosts = new double[10];
        double[] imageData = new double[784];
        int epoch = 0;
        int g = 0;
        int count = 0;
        int index = 0;
        double right = 0;
        double wrong = 0;
        double lowest = 10000;
        double previousLowest = 10000;
        double learningRate = 0.5;
        double modificationRate = 0.0002;
        System.out.println(testImages.size());
        //Collections.shuffle(trainImages);
        while (epoch < 200) {
            System.out.println("Cost: " + net.getCost());
            net.setCost(0.0);
            if (count % 500 == 0 || (count + 1) % 500 == 0) {
                Collections.shuffle(trainImages);
                holder = new ArrayList<Image>();
                for (int i = 0; i < trainImages.size(); i++) {
                    if (i < 50000) {
                        holder.add(trainImages.get(i));
                        if (holder.size() % 100 == 0) {
                            trainImageSets.add(holder);
                            g++;
                            holder = new ArrayList<Image>();
                        }
                    } else {
                        validationImages.add(trainImages.get(i));
                    }
                }
            }
            for (int i = 0; i < 500; i++) {
                //trainImageSets.size()
                for (int j = 0; j < trainImageSets.get(i).size(); j++) {
                    //trainImageSets.get(i).size()
                    imageData = new double[784];
                    for (int k = 0; k < trainImageSets.get(i).get(j).getData().length; k++) {
                        for (int l = 0; l < trainImageSets.get(i).get(j).getData()[0].length; l++) {
                            imageData[k * 28 + l] = trainImageSets.get(i).get(j).getData()[k][l];
                        }
                    }
                    net.feedForward(imageData, trainImageSets.get(i).get(j).getLabel());
                    net.backPropogation(trainImageSets.get(i).get(j).getLabel());
                    //System.out.println(net.toString2());
                    //System.out.println();
                    //System.out.println();
                    //System.out.println();
                    if (j == 99) {
                        net.AnZ(trainImageSets.get(i).get(j).getLabel());
                    }
                }
                //System.out.println(net.toString2());
                net.subtract(learningRate);
                if(epoch > 99){
                    costs[index] = net.getCost();
                    index++;
                }
                if(epoch > 99 && (i + 1) % 10 == 0){
                    lowest = 10000;
                    for(int j = 0 ; j < 10; j++){
                        if(costs[j] < lowest){
                            lowest = costs[j];
                        }
                    }
                    if(previousLowest < lowest){
                        if(learningRate > 0){
                            learningRate -= modificationRate;
                        }
                    }
                    previousLowest = lowest;
                    index = 0;

                }
                System.out.println("Count: " + count + " Cost: " + net.getCost());
                System.out.println("epoch: " + epoch);
                System.out.println("learningRate: " + learningRate);
                count++;
                System.out.println();
            }
            for (int i = 0; i < validationImages.size(); i++) {
                imageData = new double[784];
                for (int k = 0; k < validationImages.get(i).getData().length; k++) {
                    for (int l = 0; l < validationImages.get(i).getData()[0].length; l++) {
                        imageData[k * 28 + l] = validationImages.get(i).getData()[k][l];
                    }
                }
                net.feedForward(imageData, validationImages.get(i).getLabel());
                if (net.compare(validationImages.get(i).getLabel()) == true) {
                    right++;
                    System.out.println("right");
                } else {
                    wrong++;
                    System.out.println("wrong");
                }
            }
            System.out.println(right / (right + wrong));
            right = 0;
            wrong = 0;
            epoch++;
            validationImages.clear();
            trainImageSets.clear();
        }
        System.out.println("size: " + testImages.size());
        for(int i = 0; i < testImages.size(); i++){
            System.out.println("Hi");
            imageData = new double[784];
            for(int k = 0; k < testImages.get(i).getData().length; k++){
                for(int l = 0; l < testImages.get(i).getData()[0].length; l++){
                    imageData[k * 28 + l] = testImages.get(i).getData()[k][l];
                }
            }
            net.feedForward(imageData, testImages.get(i).getLabel());
            if(net.compare(testImages.get(i).getLabel()) == true){
                right++;
                System.out.println("right");
            }
            else {
                wrong++;
                System.out.println("wrong");
            }
        }
        System.out.println(right / (right + wrong));
        System.out.println("hello");


        //for(int i = 0; i < trainImageSets.size(); i++){
        //for(int j = 0; j < trainImageSets.get(i).size(); j++){
        //for(int k = 0; k < trainImageSets.get(i).get(j).getData().length; k++){
        //for(int l = 0; l < trainImageSets.get(i).get(j).getData()[0].length; l++){
        //System.out.print(trainImageSets.get(i).get(j).getData()[k][l] + ", ");
        //}
        //}
        //System.out.println();
        //}
        //}


    }
}
