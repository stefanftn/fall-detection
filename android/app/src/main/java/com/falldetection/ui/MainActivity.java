package com.falldetection.ui;

import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.falldetection.R;
import com.falldetection.api.ApiClient;
import com.falldetection.sensor.SensorCollector;

import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * MainActivity.java
 * =================
 * Jedini ekran aplikacije. Koordinira:
 *   - Spinner za izbor modela
 *   - Start/Stop dugme
 *   - SensorCollector → FeatureExtractor → ApiClient pipeline
 *   - Prikaz rezultata (status, confidence, statistike)
 *
 * Thread model:
 *   Senzori i feature ekstrakcija → SensorEventListener thread (Android)
 *   HTTP pozivi                   → background ExecutorService thread
 *   UI update                     → Main (UI) thread via Handler
 */
public class MainActivity extends AppCompatActivity
        implements SensorCollector.WindowCallback {

    // ── UI elementi ──────────────────────────────────────────────────
    private Spinner  spinnerModel;
    private Button   btnStartStop;
    private TextView tvStatus;
    private TextView tvStatusLabel;
    private TextView tvConfidence;
    private TextView tvModelUsed;
    private TextView tvInferenceMs;
    private TextView tvWindowCount;
    private TextView tvFallCount;
    private TextView tvServerUrl;
    private View     cardStatus;

    // ── Stanje ───────────────────────────────────────────────────────
    private boolean          isRunning    = false;
    private String           selectedModel = "gnb";
    private int              windowCount  = 0;
    private int              fallCount    = 0;
    private long             lastFallTime = 0;

    // ── Backend ───────────────────────────────────────────────────────
    private SensorCollector  sensorCollector;
    private ApiClient        apiClient;

    /** Single-thread executor za serijalizaciju HTTP poziva */
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    /** Handler za update UI-a sa background thread-a */
    private final Handler uiHandler = new Handler(Looper.getMainLooper());

    // ── Opcije modela ─────────────────────────────────────────────────
    private static final String[] MODEL_LABELS = {
            "Gaussian Naive Bayes", "k-Nearest Neighbors (kNN)", "MLP Neuronska Mreža"
    };
    private static final String[] MODEL_KEYS = { "gnb", "knn", "mlp" };

    // ── Vreme čuvanja FALL statusa na ekranu (ms) ─────────────────────
    private static final long FALL_DISPLAY_DURATION_MS = 4000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        bindViews();
        setupSpinner();
        setupButton();

        sensorCollector = new SensorCollector(this, this);
        apiClient       = new ApiClient();

        tvServerUrl.setText("Server: " + ApiClient.BASE_URL);
        showIdleState();
    }

    // ─────────────────────────────────────────────────────────────────
    // SETUP
    // ─────────────────────────────────────────────────────────────────

    private void bindViews() {
        spinnerModel   = findViewById(R.id.spinnerModel);
        btnStartStop   = findViewById(R.id.btnStartStop);
        tvStatus       = findViewById(R.id.tvStatus);
        tvStatusLabel  = findViewById(R.id.tvStatusLabel);
        tvConfidence   = findViewById(R.id.tvConfidence);
        tvModelUsed    = findViewById(R.id.tvModelUsed);
        tvInferenceMs  = findViewById(R.id.tvInferenceMs);
        tvWindowCount  = findViewById(R.id.tvWindowCount);
        tvFallCount    = findViewById(R.id.tvFallCount);
        tvServerUrl    = findViewById(R.id.tvServerUrl);
        cardStatus     = findViewById(R.id.cardStatus);
    }

    private void setupSpinner() {
        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this, R.layout.spinner_item, MODEL_LABELS);
        adapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
        spinnerModel.setAdapter(adapter);

        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
                selectedModel = MODEL_KEYS[pos];
            }
            @Override
            public void onNothingSelected(AdapterView<?> parent) {}
        });
    }

    private void setupButton() {
        btnStartStop.setOnClickListener(v -> {
            if (isRunning) {
                stopDetection();
            } else {
                startDetection();
            }
        });
    }

    // ─────────────────────────────────────────────────────────────────
    // START / STOP
    // ─────────────────────────────────────────────────────────────────

    private void startDetection() {
        // Proveri server u pozadini pre starta
        spinnerModel.setEnabled(false);
        btnStartStop.setEnabled(false);
        btnStartStop.setText("Proveravam server...");

        executor.submit(() -> {
            boolean healthy = apiClient.checkHealth();
            uiHandler.post(() -> {
                if (!healthy) {
                    showToast("Server nije dostupan!\nProveri IP adresu u ApiClient.java");
                    spinnerModel.setEnabled(true);
                    btnStartStop.setEnabled(true);
                    btnStartStop.setText("START");
                    return;
                }
                // Server OK — pokreni senzore
                boolean sensorsOk = sensorCollector.start();
                if (!sensorsOk) {
                    showToast("Akcelerometar nije dostupan na ovom uređaju!");
                    spinnerModel.setEnabled(true);
                    btnStartStop.setEnabled(true);
                    btnStartStop.setText("START");
                    return;
                }
                isRunning    = true;
                windowCount  = 0;
                fallCount    = 0;
                btnStartStop.setEnabled(true);
                btnStartStop.setText("STOP");
                showNormalState();
                updateCounters();
            });
        });
    }

    private void stopDetection() {
        sensorCollector.stop();
        isRunning = false;
        spinnerModel.setEnabled(true);
        btnStartStop.setText("START");
        showIdleState();
    }

    // ─────────────────────────────────────────────────────────────────
    // WINDOW CALLBACK — dolazi iz sensor thread-a
    // ─────────────────────────────────────────────────────────────────

    @Override
    public void onWindowReady(double[] features) {
        // Ne blokirati sensor thread — odmah prosleđuj u executor
        String modelToUse = selectedModel;   // thread-safe read

        executor.submit(() ->
            apiClient.sendPrediction(modelToUse, features, new ApiClient.PredictCallback() {
                @Override
                public void onSuccess(ApiClient.PredictResult result) {
                    uiHandler.post(() -> handleResult(result));
                }
                @Override
                public void onError(String errorMessage) {
                    uiHandler.post(() -> showErrorBrief(errorMessage));
                }
            })
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // UI UPDATES — sve se izvršava na Main thread-u
    // ─────────────────────────────────────────────────────────────────

    private void handleResult(ApiClient.PredictResult result) {
        windowCount++;
        if (result.isFall()) {
            fallCount++;
            lastFallTime = System.currentTimeMillis();
            showFallState(result);
            // Vrati na normalan prikaz posle FALL_DISPLAY_DURATION_MS
            uiHandler.postDelayed(this::showNormalStateIfStillRunning,
                    FALL_DISPLAY_DURATION_MS);
        } else {
            // Prikaži fall indikator dok ne istekne FALL_DISPLAY_DURATION_MS
            long elapsed = System.currentTimeMillis() - lastFallTime;
            if (elapsed >= FALL_DISPLAY_DURATION_MS) {
                showNormalState();
            }
        }

        tvConfidence.setText(String.format(Locale.US, "Confidence: %.1f%%",
                result.confidence * 100));
        tvModelUsed.setText("Model: " + result.model_used.toUpperCase());
        tvInferenceMs.setText(String.format(Locale.US, "Inference: %.1f ms",
                result.inference_ms));
        updateCounters();
    }

    private void showNormalStateIfStillRunning() {
        long elapsed = System.currentTimeMillis() - lastFallTime;
        if (elapsed >= FALL_DISPLAY_DURATION_MS && isRunning) {
            showNormalState();
        }
    }

    private void showIdleState() {
        cardStatus.setBackgroundColor(
                ContextCompat.getColor(this, R.color.status_idle));
        tvStatus.setText("⏸");
        tvStatusLabel.setText("ZAUSTAVLJENO");
        tvConfidence.setText("Confidence: —");
        tvModelUsed.setText("Model: —");
        tvInferenceMs.setText("Inference: —");
    }

    private void showNormalState() {
        cardStatus.setBackgroundColor(
                ContextCompat.getColor(this, R.color.status_ok));
        tvStatus.setText("✓");
        tvStatusLabel.setText("NORMALNO");
    }

    private void showFallState(ApiClient.PredictResult result) {
        cardStatus.setBackgroundColor(
                ContextCompat.getColor(this, R.color.status_fall));
        tvStatus.setText("⚠");
        tvStatusLabel.setText("PAD DETEKTOVAN!");
    }

    private void showErrorBrief(String msg) {
        // Ne prekidamo detekciju zbog jednog propuštenog API poziva
        tvInferenceMs.setText("Greška: timeout");
    }

    private void updateCounters() {
        tvWindowCount.setText("Prozori: " + windowCount);
        tvFallCount.setText("Padovi: " + fallCount);
    }

    private void showToast(String msg) {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show();
    }

    // ─────────────────────────────────────────────────────────────────
    // LIFECYCLE
    // ─────────────────────────────────────────────────────────────────

    @Override
    protected void onPause() {
        super.onPause();
        // Zaustavi senzore kad app ide u background (štedi bateriju)
        if (isRunning) {
            sensorCollector.stop();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Ponovo pokreni senzore ako je detekcija bila aktivna
        if (isRunning) {
            sensorCollector.start();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (isRunning) sensorCollector.stop();
        executor.shutdown();
    }
}
