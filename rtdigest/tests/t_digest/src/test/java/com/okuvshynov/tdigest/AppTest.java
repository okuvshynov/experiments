package com.okuvshynov.tdigest;

import static org.junit.Assert.assertTrue;

import com.tdunning.math.stats.TDigest;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.lang.Math;
import java.util.Scanner;
import org.junit.Test;
import java.util.ArrayList;

public class AppTest 
{
    @Test
    public void shouldAnswerWithTrue()
    {
      try {
        Scanner scanner = new Scanner(new File("../data/medium2.in"));
        PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("../logs/tdigest.log")));
        while (scanner.hasNextInt()) {
          int n = scanner.nextInt();
          double q = scanner.nextDouble() / 1000000.0;
          double expected = scanner.nextDouble();
          ArrayList<Double> vv = new ArrayList<Double>();
          for (int i = 0; i < n; i++) {
            vv.add(scanner.nextDouble());
          }

          for (int accuracy = 30; accuracy < 150; accuracy++) {
            TDigest d0 = TDigest.createMergingDigest(accuracy);
            for (double v : vv) {
              d0.add(v, 1);
            }
            out.format("tdigest,%d,%f,%f\n", d0.centroids().size(), q, Math.abs(expected - d0.quantile(q)));
          }
        }
        out.close();
      } catch (Exception e) {
      }
    }
}
