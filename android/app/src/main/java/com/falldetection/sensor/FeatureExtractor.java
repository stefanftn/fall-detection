package com.falldetection.sensor;

/**
 * FeatureExtractor.java
 * =====================
 * Računa 17 obeležja iz jednog sliding window prozora.
 *
 * MORA biti identičan Python implementaciji u features.py (extract_features()).
 * Svako odstupanje u formuli daje pogrešne predikcije čak i sa savršenim modelom.
 *
 * Redosled feature-a (obavezan, mora odgovarati feature_names.json sa servera):
 *   0  vm_mean       — srednja VM akcelerometra
 *   1  vm_max        — maksimalna VM (key feature za pad)
 *   2  vm_std        — std devijacija VM
 *   3  vm_range      — max - min VM
 *   4  vm_energy     — energija: sum(vm^2) / N
 *   5  vm_zcr        — zero crossing rate VM oko srednje vrednosti
 *   6  ax_std        — std X ose
 *   7  ay_std        — std Y ose
 *   8  az_std        — std Z ose
 *   9  ax_max        — max |X|
 *  10  ay_max        — max |Y|
 *  11  az_max        — max |Z|
 *  12  gyro_vm_mean  — srednja VM žiroskopa
 *  13  gyro_vm_max   — maksimalna VM žiroskopa
 *  14  gyro_vm_std   — std VM žiroskopa
 *  15  sma           — Signal Magnitude Area
 *  16  tilt_post     — srednja Z u poslednjoj 1/3 prozora
 *
 * Ulazni podaci su u fizičkim jedinicama (g za akcelerometar, ADC za žiroskop)
 * — konverzija je obavljena u SensorCollector pre poziva ove klase.
 */
public class FeatureExtractor {

    public static final int N_FEATURES = 17;

    /**
     * Računa 17 obeležja iz jednog prozora.
     *
     * @param ax  niz X vrednosti akcelerometra (u g), dužina = WINDOW_SIZE
     * @param ay  niz Y vrednosti akcelerometra (u g)
     * @param az  niz Z vrednosti akcelerometra (u g)
     * @param gx  niz X vrednosti žiroskopa (ADC sirove vrednosti)
     * @param gy  niz Y vrednosti žiroskopa
     * @param gz  niz Z vrednosti žiroskopa
     * @return    double[] sa 17 feature vrednosti u definisanom redosledu
     */
    public static double[] extract(double[] ax, double[] ay, double[] az,
                                   double[] gx, double[] gy, double[] gz) {
        int n = ax.length;

        // ── Vector Magnitude (VM) po uzorku ──────────────────────────
        double[] vm     = new double[n];
        double[] gyroVm = new double[n];
        for (int i = 0; i < n; i++) {
            vm[i]     = Math.sqrt(ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i]);
            gyroVm[i] = Math.sqrt(gx[i]*gx[i] + gy[i]*gy[i] + gz[i]*gz[i]);
        }

        // ── VM statistike ─────────────────────────────────────────────
        double vmMean   = mean(vm);
        double vmMax    = max(vm);
        double vmMin    = min(vm);
        double vmStd    = std(vm, vmMean);
        double vmRange  = vmMax - vmMin;
        double vmEnergy = energy(vm, n);
        double vmZcr    = zcr(vm, vmMean, n);

        // ── Per-osa statistike (ADXL345) ──────────────────────────────
        double axStd = std(ax, mean(ax));
        double ayStd = std(ay, mean(ay));
        double azStd = std(az, mean(az));
        double axMax = maxAbs(ax);
        double ayMax = maxAbs(ay);
        double azMax = maxAbs(az);

        // ── Žiroskop VM statistike ─────────────────────────────────────
        double gyroVmMean = mean(gyroVm);
        double gyroVmMax  = max(gyroVm);
        double gyroVmStd  = std(gyroVm, gyroVmMean);

        // ── Signal Magnitude Area ──────────────────────────────────────
        double sma = sma(ax, ay, az, n);

        // ── Tilt post-impact: srednja Z u poslednjoj 1/3 prozora ──────
        int postStart = n * 2 / 3;
        double tiltPost = meanRange(az, postStart, n);

        return new double[] {
            vmMean, vmMax, vmStd, vmRange, vmEnergy, vmZcr,
            axStd, ayStd, azStd, axMax, ayMax, azMax,
            gyroVmMean, gyroVmMax, gyroVmStd,
            sma, tiltPost
        };
    }

    // ─────────────────────────────────────────────────────────────────
    // POMOĆNE METODE  (identične Python numpy operacijama)
    // ─────────────────────────────────────────────────────────────────

    private static double mean(double[] a) {
        double s = 0;
        for (double v : a) s += v;
        return s / a.length;
    }

    private static double meanRange(double[] a, int from, int to) {
        double s = 0;
        for (int i = from; i < to; i++) s += a[i];
        return s / (to - from);
    }

    private static double max(double[] a) {
        double m = a[0];
        for (double v : a) if (v > m) m = v;
        return m;
    }

    private static double min(double[] a) {
        double m = a[0];
        for (double v : a) if (v < m) m = v;
        return m;
    }

    private static double maxAbs(double[] a) {
        double m = 0;
        for (double v : a) if (Math.abs(v) > m) m = Math.abs(v);
        return m;
    }

    /** Populaciona std devijacija (ddof=0), identično numpy.std() */
    private static double std(double[] a, double mu) {
        double s = 0;
        for (double v : a) s += (v - mu) * (v - mu);
        return Math.sqrt(s / a.length);
    }

    /** Energija: sum(v^2) / N — identično Python implementaciji */
    private static double energy(double[] a, int n) {
        double s = 0;
        for (double v : a) s += v * v;
        return s / n;
    }

    /**
     * Zero Crossing Rate centiran oko srednje vrednosti.
     * Python: np.sum(np.abs(np.diff(np.sign(vm - vm.mean())))) / (2 * N)
     * Svaki prelaz nule (promena znaka) se broji jednom.
     */
    private static double zcr(double[] a, double mu, int n) {
        int crossings = 0;
        int prevSign  = sign(a[0] - mu);
        for (int i = 1; i < n; i++) {
            int curSign = sign(a[i] - mu);
            if (curSign != prevSign && curSign != 0) {
                crossings++;
                prevSign = curSign;
            }
        }
        return (double) crossings / (2.0 * n);
    }

    private static int sign(double v) {
        if (v > 0) return 1;
        if (v < 0) return -1;
        return 0;
    }

    /** SMA = (sum|x| + sum|y| + sum|z|) / N */
    private static double sma(double[] ax, double[] ay, double[] az, int n) {
        double s = 0;
        for (int i = 0; i < n; i++)
            s += Math.abs(ax[i]) + Math.abs(ay[i]) + Math.abs(az[i]);
        return s / n;
    }
}
