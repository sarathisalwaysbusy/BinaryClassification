//C:\Users\sarath\Desktop\Selto Assignment\A1\train.txt
/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml2_a1;

import Jama.Matrix;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Scanner;
import static ml2_a1.A1Q2.tn;
//C:\Users\sarath\Desktop\4-2\SelTo\A1\train.txt

/**
 * Contains the main method and other important static instance variables in fisher discrimination.
 * @author sarath
 */
public class A1Q1 {
    
    static List<Data> dataC0, dataC1, testData, trainingData;
    static X meanVector0, meanVector1;
    static Matrix meanMatrix0, meanMatrix1, SwMatrix, wMatrix;
    static int n, n0, n1, tp=0, tn=0, fp=0, fn=0;
    static double threshold, modW;
    
    
    public static void main(String [] args)
    {
        initData();
        calculateW();
        
        //get threshold
        transformData();
        test();
        report();
    }
    
    static void report()
    {
        System.out.println("");
        System.out.println("Total correct classificaitons = " + (tn+tp));
        System.out.println("Total incorrect classifications = "+ (fn+fp));
        System.out.println("");
        System.out.println("Confusion Matrix:-");
        System.out.println("  Predicted class  ");
        System.out.println("   P      N   ");
        System.out.println(" P "+ tp + "    "+ fn);
        System.out.println("");
        System.out.println(" N "+ fp + "    "+ tn);
        System.out.println("");
        
        double accuracy = (double) (tp + tn)* 100 /(double) (tp+fp+tn+fn) ;
        double precision = (double)tp * 100/(double)(tp+fp);
        double recall = (double) tp * 100 /(double) (tp+fn);
        
        System.out.println("Accuracy = " + accuracy+ "%");
        System.out.println("Recall = " + recall + "%");
        System.out.println("Precision = " + precision + "%");
        System.out.println("F1 Score = " + 2*recall*precision/(recall + precision) + "%");
        
    }
    /**
     * initializes the testing data and tests the model.
     */
    static void test()
    {
        initTestingData();
        for(Data d : testData)
        {
            double[][] transformedArray = wMatrix.transpose().times(new Matrix(Utils2.getMatrixArray(d.x))).getArray();
            double transformedX = transformedArray[0][0]/modW;
            int myClassification;
            if(transformedX < threshold)
            {
                myClassification = 0;
            }
            else
            {
                myClassification = 1;
            }
            
            
            if(d.target == 1)
            {
                if(myClassification == 1)
                {
                    tp++;
                }
                else
                {
                    fn++;
                }
            }
            else 
            {
                if(myClassification == 1)
                {
                    fp++;
                }
                else
                {
                    tn++;
                }
            }
//            System.out.println("test x = "+transformedX + ", threshold = "+ threshold + "     "
//                    + "actual class = " + d.target + ",    my classification = " + myClassification);
        }
    }
    
    /**
     * obtains the testing data from a specified file.
     */
    static void initTestingData()
    {
        testData = new ArrayList<>();
        
        System.out.println("Specify the directory of the test.txt file");
        Scanner in = new Scanner(System.in);
        
        BufferedReader br = null;
        try {
            
            br = new BufferedReader(new FileReader(new File(in.nextLine())));
            
            String line = "";
            while((line = br.readLine()) != null)
            {
                String[] data = line.split(",");
                
                
                testData.add(new Data(data[0], data[1], data[2], data[3], data[4]));
                
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
        
    }
    
    /**
     * transforms the training data into 1 dimension using wTx and calculates
     * the value of threshold.
     */
    static void transformData()
    {
        List<TransformedX> transformedList = new ArrayList<TransformedX>();
        
        int j=0;
        modW = Utils2.getMod(wMatrix.getArray());
        
        // for each data, project it onto the line;
        for(Data d:trainingData)
        {
            Matrix x = new Matrix(Utils2.getMatrixArray(d.x));
            double[][] array = wMatrix.transpose().times(x).getArray();
            
            //dividing by |w|.
            transformedList.add(new TransformedX(array[0][0]/modW,d.target)) ;
        }
        
        transformedList.sort(Comparator.comparing(TransformedX::getX));
        
        // calculating entropy..
        double entropy = Utils2.calculateEntropy(transformedList, transformedList.get(0).x);
        
        // find lowest entropy point
        for(TransformedX x:transformedList)
        {    
            if(Utils2.calculateEntropy(transformedList, x.x)< entropy )
            {
                entropy = Utils2.calculateEntropy(transformedList, x.x);
                threshold = x.x;
            }
        }
        
        //System.out.println("Threshold = " + threshold);
        
    }
    
    /**
     * calculates w;
     */
    static void calculateW()
    {
        n0 = dataC0.size();
        n1 = dataC1.size();
        n = n0 + n1;
        
        meanVector0 = new X();
        meanVector1 = new X();
        
        for(Data d:dataC0)
        {
            Utils2.vectorAddition(meanVector0, d.x);
        }
        Utils2.vectorDivision(meanVector0, n0);
        
        for(Data d:dataC1)
        {
            Utils2.vectorAddition(meanVector1, d.x);
        }
        Utils2.vectorDivision(meanVector1, n1);
        
        //m1
        meanMatrix0 = new Matrix(Utils2.getMatrixArray(meanVector0));
        //m2
        meanMatrix1 = new Matrix(Utils2.getMatrixArray(meanVector1));
        
        // m2-m1
        Matrix meanDiffMatrix = meanMatrix1.minus(meanMatrix0);
        
        // init Sw
        double [][] initSw = new double[4][4];
        for(int i=0; i<4; i++)
        {
            for(int j =0; j<4; j++)
            {
                initSw[i][j] = 0.0;
            }
        }
        
        SwMatrix = new Matrix(initSw);
        
        for(Data d:dataC0)
        {
            X meanDiffvector = Utils2.returnVectorSubtaction(d.x, meanVector0);
            Matrix temp = new Matrix(Utils2.getMatrixArray(meanDiffvector));
            
            Matrix toAdd = temp.times(temp.transpose());
            // sw += sum(x-m1)(x-m1)T   x in c1
            SwMatrix = SwMatrix.plus(toAdd);
        }
        
        for(Data d:dataC1)
        {
            X meanDiffvector = Utils2.returnVectorSubtaction(d.x, meanVector1);
            Matrix temp = new Matrix(Utils2.getMatrixArray(meanDiffvector));
            
            Matrix toAdd = temp.times(temp.transpose());
            // sw += sum(x-m1)(x-m1)T   x in c2
            SwMatrix = SwMatrix.plus(toAdd);
        }
        
        // w = Sw-1 (m2-m1)
        wMatrix = SwMatrix.inverse().times(meanDiffMatrix);
    }
    
    /**
     * Obtains the training data from a specified file, and initializes dataC1,
     * dataC2 and trainingData arraylists.
     */
    static void initData()
    {
        dataC1 = new ArrayList<>();
        dataC0 = new ArrayList<>();
        trainingData = new ArrayList<>();
        
        System.out.println("Specify the directory of the train.txt file");
        Scanner in = new Scanner(System.in);
        
        BufferedReader br = null;
        try {
            
            br = new BufferedReader(new FileReader(new File(in.nextLine())));
            
            String line = "";
            while((line = br.readLine()) != null)
            {
                String[] data = line.split(",");
                if(data[data.length-1].equals("0"))
                {
                    dataC0.add(new Data(data[0], data[1], data[2], data[3], data[4]));
                    // add it to the total training data
                    trainingData.add(dataC0.get(dataC0.size()-1));
                }
                else
                {
                    dataC1.add(new Data(data[0], data[1], data[2], data[3], data[4]));
                    // add it to the total training data
                    trainingData.add(dataC1.get(dataC1.size()-1));
                }
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
}
