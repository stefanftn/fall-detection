package com.falldetection.sensor;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

/**
 * SensorCollector.java
 * ====================
 * Registruje akcelerometar i žiroskop, prikuplja uzorke u ring buffer
 * i okida feature ekstrakciju na svakih STEP_SAMPLES novih uzoraka.
 *
 * Parametri odgovaraju Python pipeline-u:
 *   FS             = 200 Hz
 *   WINDOW_SAMPLES = 600  (3 sekunde × 200 Hz)
 *   STEP_SAMPLES   = 300  (50% overlap)
 *   ADXL345_SCALE  = 0.00390625  (ADC → g konverzija)
 *
 * NAPOMENA o Android senzorima:
 *   Android TYPE_ACCELEROMETER vraća vrednosti u m/s².
 *   Deljenjem sa 9.80665 dobijamo g jedinice koje model očekuje.
 *   Žiroskop (TYPE_GYROSCOPE) vraća rad/s — šaljemo sirove vrednosti
 *   pošto model nije fitovan na fizički konvertovane gyro vrednosti
 *   (SisFall dataset čuva ADC vrednosti žiroskopa).
 *
 * Callback interface:
 *   onWindowReady(features) — poziva se svaki put kad je prozor spreman
 */
public class SensorCollector implements SensorEventListener {

    private static final String TAG = "SensorCollector";

    // Identično Python konstantama
    public static final int    WINDOW_SAMPLES = 600;
    public static final int    STEP_SAMPLES   = 300;
    private static final float G_TO_MS2       = 9.80665f;   // 1g u m/s²

    // Ring buffers za akcelerometar i žiroskop
    private final double[] bufAx, bufAy, bufAz;
    private final double[] bufGx, bufGy, bufGz;
    private int writePos = 0;       // pozicija pisanja u ring buffer-u
    private int filled   = 0;       // koliko uzoraka ima u buffer-u
    private int stepCounter = 0;    // broji uzorke od poslednjeg okidanja

    private final SensorManager    sensorManager;
    private       Sensor           accelerometer;
    private       Sensor           gyroscope;
    private final WindowCallback   callback;

    // Poslednje poznate gyro vrednosti (sinhronizacija sa accel eventima)
    private volatile double lastGx = 0, lastGy = 0, lastGz = 0;

    public interface WindowCallback {
        /** Poziva se iz pozadinskog thread-a kada je prozor spreman. */
        void onWindowReady(double[] features);
    }

    public SensorCollector(Context context, WindowCallback callback) {
        this.callback      = callback;
        this.sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);

        bufAx = new double[WINDOW_SAMPLES];
        bufAy = new double[WINDOW_SAMPLES];
        bufAz = new double[WINDOW_SAMPLES];
        bufGx = new double[WINDOW_SAMPLES];
        bufGy = new double[WINDOW_SAMPLES];
        bufGz = new double[WINDOW_SAMPLES];
    }

    /** Pokreće prikupljanje. Frekvencija: SENSOR_DELAY_FASTEST ≈ 200 Hz na većini uređaja. */
    public boolean start() {
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscope     = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);

        if (accelerometer == null) {
            Log.e(TAG, "Akcelerometar nije dostupan na ovom uređaju!");
            return false;
        }

        sensorManager.registerListener(this, accelerometer,
                SensorManager.SENSOR_DELAY_FASTEST);

        if (gyroscope != null) {
            sensorManager.registerListener(this, gyroscope,
                    SensorManager.SENSOR_DELAY_FASTEST);
        } else {
            Log.w(TAG, "Žiroskop nije dostupan — gyro obeležja biće 0.");
        }

        // Reset stanja
        writePos    = 0;
        filled      = 0;
        stepCounter = 0;
        lastGx = lastGy = lastGz = 0;

        Log.i(TAG, "Senzori pokrenuti. Window=" + WINDOW_SAMPLES
                + " Step=" + STEP_SAMPLES);
        return true;
    }

    /** Zaustavlja prikupljanje i otregistruje listenere. */
    public void stop() {
        sensorManager.unregisterListener(this);
        Log.i(TAG, "Senzori zaustavljeni.");
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            // Sačuvaj poslednje gyro vrednosti (u rad/s)
            lastGx = event.values[0];
            lastGy = event.values[1];
            lastGz = event.values[2];
            return;
        }

        if (event.sensor.getType() != Sensor.TYPE_ACCELEROMETER) return;

        // Konvertuj m/s² → g (model je treniran na g vrednostima)
        double ax = event.values[0] / G_TO_MS2;
        double ay = event.values[1] / G_TO_MS2;
        double az = event.values[2] / G_TO_MS2;

        // Upiši u ring buffer (kružna prepisivanje)
        bufAx[writePos] = ax;
        bufAy[writePos] = ay;
        bufAz[writePos] = az;
        bufGx[writePos] = lastGx;
        bufGy[writePos] = lastGy;
        bufGz[writePos] = lastGz;

        writePos = (writePos + 1) % WINDOW_SAMPLES;
        if (filled < WINDOW_SAMPLES) filled++;

        stepCounter++;

        // Okini feature ekstrakciju kada imamo pun prozor i STEP_SAMPLES novih uzoraka
        if (filled >= WINDOW_SAMPLES && stepCounter >= STEP_SAMPLES) {
            stepCounter = 0;
            triggerExtraction();
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Nije potrebno za ovaj slučaj
    }

    /** Kopira ring buffer u linearne nizove i računa feature-e. */
    private void triggerExtraction() {
        double[] ax = new double[WINDOW_SAMPLES];
        double[] ay = new double[WINDOW_SAMPLES];
        double[] az = new double[WINDOW_SAMPLES];
        double[] gx = new double[WINDOW_SAMPLES];
        double[] gy = new double[WINDOW_SAMPLES];
        double[] gz = new double[WINDOW_SAMPLES];

        // Rasporedi ring buffer u hronološki redosled
        // writePos pokazuje na najstariji uzorak u punom bufferu
        for (int i = 0; i < WINDOW_SAMPLES; i++) {
            int idx = (writePos + i) % WINDOW_SAMPLES;
            ax[i] = bufAx[idx];
            ay[i] = bufAy[idx];
            az[i] = bufAz[idx];
            gx[i] = bufGx[idx];
            gy[i] = bufGy[idx];
            gz[i] = bufGz[idx];
        }

        double[] features = FeatureExtractor.extract(ax, ay, az, gx, gy, gz);
        callback.onWindowReady(features);
    }
}
