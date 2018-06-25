/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml2_a1;

/**
 * Used to capture the training and testing data.
 * @author sarath
 */
public class Data {
    X x;
    int target;

    // constructor #1
    public Data(String x1, String x2, String x3, String x4, String target) {
        
        this.x = new X(x1, x2, x3, x4);
        this.target = Integer.parseInt(target);
    }
    // constructor #2
    public Data(double x1, double x2, double x3, double x4, int target)
    {
        x = new X(x1, x2, x3, x4);
        this.target = target;
    }
    
    
}
