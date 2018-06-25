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
import java.util.List;
import java.util.Scanner;
//C:\Users\sarath\Desktop\4-2\SelTo\A1\train.txt

/**
 *
 * @author sarath
 */
public class A1Q2 {

    /**
     * This is the class containing the main method.
     * @param args the command line arguments
     */
    static List<Data> dataC0, dataC1, testData;
    static double pC0, pC1;
    static X mean0, mean1;
    
    static Matrix S0, S1, S;
    
    static int correct=0, wrong=0, n, n0, n1;
    static int tp=0, tn=0, fp=0, fn=0;
    
    public static void main(String[] args) {
        initData();
        initPrior();
        test();
        report();
    }
    /**
     * Generates a report of
     */
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
    static void test()
    {
        initTestingData();
        for(Data d : testData)
        {
            double prob0 = Utils2.getGaussianProb(mean0, S, d.x);
            double prob1 = Utils2.getGaussianProb(mean1, S, d.x);
            
            int classification;
            if(prob0>prob1)
            {
                classification = 0;
            }
            else
            {
                classification = 1;
            }
            
            if(d.target == classification)
            {
                if(classification==1)
                {
                    tp++;
                }
                else
                {
                    tn++;
                }
                //System.out.println("correct");
                ++correct;
            }
            else
            {
                if(classification == 1)
                {
                    fp++;
                }
                else
                {
                    fn++;
                }
                //System.out.println("wrong!");
                ++wrong;
            }
            //System.out.println("logistic sigmoid = " + getSigmoid(d.x));
        }
    }
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
    
    static void initPrior()
    {
        n0 = dataC0.size(); n1 = dataC1.size();
        n = n0 + n1;
        
        // initializing the prior class probabilities;
        pC0 = n0 / n;
        pC1 = n1 / n;
        
        // initializing the class prior distributions;
        mean0 = new X();
        mean1 = new X();
        
        for(Data d : dataC0)
        {
            Utils2.vectorAddition(mean0, d.x);
        }
        Utils2.vectorDivision(mean0, n0);
        
        for(Data d : dataC1)
        {
            Utils2.vectorAddition(mean1, d.x);
        }
        Utils2.vectorDivision(mean1, n0);
        
        ////////////////////////////////////////////////////////
        
        // TODO : no need for another double array for initializing S1;
        
        // calculation of S1 and S2;
        double[][] tempS0 = new double[4][4];
        double[][] tempS1 = new double[4][4];
        initMatrix(tempS0, 4, 4);
        initMatrix(tempS1, 4, 4);
        
        S0 = new Matrix(tempS0);
        S1 = new Matrix(tempS1);
        
        S = new Matrix(tempS0);
        
        for(Data d : dataC0)
        {
            X meanDiffVector = Utils2.returnVectorSubtaction(d.x, mean0);
            
            Matrix meanDiffMatrix = new Matrix( Utils2.getMatrixArray(meanDiffVector) );
            //Matrix meanDiffTransMatrix = new Matrix(Utils2.getMatrixTransArray(meanDiffVector));
            Matrix toAdd = meanDiffMatrix.times(meanDiffMatrix.transpose());
            //System.out.println("toAdd = " + Arrays.deepToString(toAdd.getArray()));
            S0 = S0.plus(toAdd);
            //System.out.println("S0 = " + Arrays.deepToString(S0.getArray()));
        }
        S0 = S0.times((1.0/n0));
        
        
        for(Data d : dataC1)
        {
            X meanDiffVector = Utils2.returnVectorSubtaction(d.x, mean1);
            
            Matrix meanDiffMatrix = new Matrix( Utils2.getMatrixArray(meanDiffVector) );
            //Matrix meanDiffTransMatrix = new Matrix(Utils2.getMatrixTransArray(meanDiffVector));
            //
            Matrix toAdd = meanDiffMatrix.times(meanDiffMatrix.transpose());
            S1 = S1.plus(toAdd);
        }
        S1 = S1.times(1.0/n1);
        
        Matrix toAddS1 = S0.times((double)n0/n);
        Matrix toAddS2 = S1.times((double)n1/n);
        
        S = S.plus(toAddS1);
        S = S.plus(toAddS2);
        
//        for(Data d: dataC0)
//        {
//            S0 += Utils2.vectorSquare(Utils2.returnVectorSubtaction(d.x, mean0));
//        }
//        for(Data d: dataC1)
//        {
//            S1 += Utils2.vectorSquare(Utils2.returnVectorSubtaction(d.x, mean1));
//        }
//        
//        S0/=n0;
//        S1/=n1;
//        
//        // S = N1*S1/N + N2*S2/N;
//        S=(n0*S0/n) + (n1*S1/n);
//        //System.out.println("S = "+ S);
    }
    
    // initializes the matrix
    static void initMatrix(double [][] mat, int n, int m)
    {
        for(int i=0; i<n; i++)
        {
            for(int j=0; j<n; j++)
            {
                mat[i][j] = 0;
            }
        }
    }
    
    static void myPrint(double [][] mat, int n, int m)
    {
        for(double[] row:mat)
        {
            for(double col:row)
            {
                System.out.println(col+" ");
            }
            System.out.println("");
        }
    }
    static void myPrint(X x)
    {
        System.out.println(x.x1 + " " + x.x2 + " " + x.x3 + " "+ x.x4);
    }
    
    static void myPrint(List<Data> data)
    {
        for(Data d:data)
        {
            System.out.println(d.x.x1 + " " + d.x.x2 + " " + d.x.x3 + " " +
                    d.x.x4 + " " + d.target);
        }
    }
    static void initData()
    {
        dataC1 = new ArrayList<>();
        dataC0 = new ArrayList<>();
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
                }
                else
                {
                    dataC1.add(new Data(data[0], data[1], data[2], data[3], data[4]));
                }
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    // returns the 
    static double getSigmoid(X x)
    {
        double a = ((Utils2.getGaussianProb(mean0, S, x) * n0)/(Utils2.getGaussianProb(mean1, S, x) * n1));
        return Utils2.logisticSigmoid(a);
        
    }
}
