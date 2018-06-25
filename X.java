/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package ml2_a1;

/**
 * Class used to store the input vector.
 * @author sarath
 */
public class X{
    double x1, x2, x3, x4;

    // constructor #1
    public X(String x1, String x2, String x3, String x4) {
        this.x1 = Float.parseFloat(x1);
        this.x2 = Float.parseFloat(x2);
        this.x3 = Float.parseFloat(x3);
        this.x4 = Float.parseFloat(x4);
    }

    // constructor #1
    public X(double x1, double x2, double x3, double x4) {
        this.x1 = x1;
        this.x2 = x2;
        this.x3 = x3;
        this.x4 = x4;
    }
    
    
    public X() {
        x1 = x2 = x3 = x4  = 0;
    }
}
