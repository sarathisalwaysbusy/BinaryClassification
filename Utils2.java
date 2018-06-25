/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml2_a1;

import Jama.Matrix;
import java.util.Arrays;
import java.util.List;

/**
 * A static class with utility methods that are helpful for calculations.
 * @author sarath
 */
public class Utils2 {
    
    /**
     * returns the value 1/( 1+e^(-a) ), which is a logistic sigmoid function 
     * of the input double value- a.
     * @param a
     * @return 
     */
    static double logisticSigmoid(double a)
    {
        return 1/(1+Math.exp(-a));
    }
    /**
     * sets sum = sum + operand
     * @param sum
     * @param operand 
     */
    static void vectorAddition(X sum, X operand)
    {
        sum.x1 += operand.x1;
        sum.x2 += operand.x2;
        sum.x3 += operand.x3;
        sum.x4 += operand.x4;
    }
    
    /**
     * Returns the a new vector = @param a - @param b;
     * @param a first vector
     * @param b second vector
     * @return a-b
     */
    static X returnVectorSubtaction(X a, X b)
    {
        X c = new X();
        c.x1 = a.x1 - b.x1;
        c.x2 = a.x2 - b.x2;
        c.x3 = a.x3 - b.x3;
        c.x4 = a.x4 - b.x4;
        
        return c;
    }
    
    /**
     * returns the scalar quantity aT * a;
     * @param a the input vector
     * @return aT * a;
     */
    static double vectorSquare(X a)
    {
        double product=0;
        product = (a.x1*a.x1) + (a.x2*a.x2) + (a.x3*a.x3) + (a.x4*a.x4);
        
        return product;
    }
    
    /**
     * Divides the input @param ans with the input @param amt.
     * @param ans the vector whose value needs to be decreased.
     * @param amt the factor by which the division should occur.
     */
    static void vectorDivision(X ans, int amt)
    {
        ans.x1 /= amt;
        ans.x2 /= amt;
        ans.x3 /= amt;
        ans.x4 /= amt;
    }
    
//    /**
//     * 
//     * @param x1
//     * @param x2
//     * @param x3
//     * @param x4
//     * @return 
//     */
//    static double [] getMatrixArray(double x1, double x2, double x3, double x4)
//    {
//        double [] temp = new double[4];
//        temp[0] = x1;
//        temp[1] = x2;
//        temp[2] = x3;
//        temp[3] = x4;
//        
//        return temp;
//    }
    
    /**
     * Returns a 4X1 double array representing the vector x
     * @param x The input vector of type X to be transformed into the 2D array.
     * @return The transformed 2D array.
     */
    static double [][] getMatrixArray(X x)
    {
        double [][] temp = new double[4][1];
        temp[0][0] = x.x1;
        temp[1][0] = x.x2;
        temp[2][0] = x.x3;
        temp[3][0] = x.x4;
        
        return temp;
    }
    
    /**
     * Returns a 1X4 double array representing the vector xT (read as- x transpose).
     * @param x The input vector of type X;
     * @return The transformed 2D array
     */
    static double [][] getMatrixTransArray(X x)
    {
        double [][] temp = new double[1][4];
        temp[0][0] = x.x1;
        temp[0][1] = x.x2;
        temp[0][2] = x.x3;
        temp[0][3] = x.x4;
        
        return temp;
    }
    
    /**
     * Returns the modulous value of the input row/column matrix.
     * @param matrix A two dimensional matrix representing a one-dimensional w vector.
     * @return |w|
     */
    static double getMod(double [][] matrix)
    {
        double mod = 0;
        for(int i=0; i<matrix.length; i++)
        {
            for(int j=0; j<matrix[0].length; j++)
            {
                mod+= Math.pow(matrix[i][j], 2);
            }
        }
        mod = Math.sqrt(mod);
        return mod;
    }
    
    /**
     * Calculates the entropy of a given dividing point (transformed) with respect to the given 
     * list of transformed input vectors.
     * @param transformedList The list of points.
     * @param x The dividing point
     * @return The Shanons entropy;
     */
    static double calculateEntropy(List<TransformedX> transformedList, double x)
    {
        // a classification is class 1 when x >= threshold.
        
        // n0 --> number of points on + ve side which are negative
        // n1 --> number  of points on - ve side which are positive
        double n1=0, n0=0;
        double total1 = 0, total0 = 0;
        
        for(TransformedX transformedX: transformedList)
        {
            // if the actual classificaiton is 1
            if(transformedX.classification == 1 )
            {
                // if my classification is 0
                if(transformedX.x < x)
                {
                    n1++;
                }
                total1++;
            }
            else    // if actual classification is 0
            {
                // if my classification is 1
                if(transformedX.x >= x)
                {
                    n0++;
                }
                total0 ++;
            }
        }
        double p0 = n0/total0, p1 = n1/total1;
        
        double entropy = (-1*p0*Math.log(p0)) + (-1*p0*Math.log(p1));
        return entropy;
    }
    /**
    *@return the prior gaussian probability of the input vector
    * 
    *@param mean is the mean vector for the gaussian distribution.
    *@param S is the constant which when multiplied with an identity matrix 
    * gives the variance matrix
    * @param x is the input vector whose prior probability needs to be found
    **/
    static double getGaussianProb(X mean, Matrix S, X x)
    {
        // exp = -1/2 (x-mean)T S^-1 (x-mean)
        double exp = 0;
        
        //meanDiffMatrix --> (x-mean)
        X meanDiffVector = Utils2.returnVectorSubtaction(x, mean);
        Matrix meanDiffMatrix = new Matrix(Utils2.getMatrixArray(meanDiffVector));
        
        //intermmediateMatrix--> (x-mean)T S^-1
        Matrix intermmediateMatrix = meanDiffMatrix.transpose().times(S.inverse());
        Matrix exponentMatrix = intermmediateMatrix.times(meanDiffMatrix);

//        System.out.println("exponet Matrix = ");
//        System.out.println(Arrays.deepToString(exponentMatrix.getArray())) ;
        
        double[][] expArray = exponentMatrix.getArray();
        exp = expArray[0][0];
        exp *= -0.5;
        

        double prob = (1/ ( (Math.pow( (2 * Math.PI), 2) ) * Math.sqrt(S.det()) ) );
        //System.out.println("1/*** = "+ prob);
        prob *= Math.pow(Math.E, exp);
        
        return (double)prob;
    }
}
