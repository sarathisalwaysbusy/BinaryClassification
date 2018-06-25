/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml2_a1;

import java.util.Comparator;

/**
 * Class used to store the projected value of the input vector as well as its true classification.
 * @author sarath
 */
public class TransformedX implements Comparator<TransformedX>{
    double x;
    int classification;

    @Override
    public int compare(TransformedX o1, TransformedX o2) {
        if(o1.x > o2.x)
        {
            return 1;
        }
        else
            return 0;
    }

    public double getX() {
        return x;
    }

    
    public TransformedX() {
        x = 0.0;
        classification = 0;
    }

    public TransformedX(double x, int classification) {
        this.x = x;
        this.classification = classification;
    }
    
    
}
