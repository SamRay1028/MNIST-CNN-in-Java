import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class DataReader {
    private final int rows = 28;
    private final int cols = 28;

    public ArrayList<Image> readData(String path){
        ArrayList<Image> images = new ArrayList<>();
        try(BufferedReader dataReader = new BufferedReader(new FileReader(path))){
            String line;
            while((line = dataReader.readLine()) != null){
                String[] lineItems = line.split(",");
                double[][] data = new double[rows][cols];
                int label = Integer.parseInt(lineItems[0]);
                int i = 1;
                for(int row = 0; row < rows; row++){
                    for(int col = 0; col < cols; col++){
                        data[row][col] = (double) Integer.parseInt(lineItems[i]);
                        i++;
                    }
                }
                images.add(new Image(data, label));
            }

        }
        catch(Exception e){

        }

        return images;
    }

}

