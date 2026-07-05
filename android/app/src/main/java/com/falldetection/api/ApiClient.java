package com.falldetection.api;

import android.util.Log;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

/**
 * ApiClient.java
 * ==============
 * Šalje POST /predict zahtev FastAPI backendu i parsira odgovor.
 *
 * Konfiguracija servera:
 *   BASE_URL — IP adresa i port laptopa/servera na istoj WiFi mreži.
 *   Primer: "http://192.168.1.100:8000"
 *   Pronađi IP adresu laptopa sa: ip addr (Linux) ili ipconfig (Windows).
 *
 * Pozivi se uvek vrše iz pozadinskog thread-a (AsyncTask / Thread).
 * Nikada ne zovi sendPrediction() iz UI threada.
 */
public class ApiClient {

    private static final String TAG = "ApiClient";

    // ── PROMENI OVU ADRESU na IP svog laptopa ──────────────────────
    public static final String BASE_URL = "http://10.113.72.161:8000";
    // ───────────────────────────────────────────────────────────────

    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    private final OkHttpClient client;
    private final Gson         gson;
    private final String       baseUrl;

    /** Rezultat predikcije primljen sa servera. */
    public static class PredictResult {
        public int    label;           // 0 = ADL, 1 = FALL
        public String label_text;      // "ADL" ili "FALL"
        public double confidence;      // verovatnoća pada [0.0–1.0]
        public String model_used;      // "gnb", "knn" ili "mlp"
        public double inference_ms;    // vreme inference-a na serveru

        public boolean isFall() { return label == 1; }
    }

    /** Callback za rezultat (poziva se iz pozadinskog thread-a). */
    public interface PredictCallback {
        void onSuccess(PredictResult result);
        void onError(String errorMessage);
    }

    public ApiClient() {
        this(BASE_URL);
    }

    public ApiClient(String baseUrl) {
        this.baseUrl = baseUrl;
        this.gson    = new Gson();
        this.client  = new OkHttpClient.Builder()
                .connectTimeout(3,  TimeUnit.SECONDS)   // kratko — real-time zahtev
                .readTimeout(5,     TimeUnit.SECONDS)
                .writeTimeout(3,    TimeUnit.SECONDS)
                .build();
    }

    /**
     * Šalje feature vektor backendu i vraća predikciju.
     * BLOKIRA pozivajući thread — koristiti iz background thread-a.
     *
     * @param modelName  "gnb", "knn" ili "mlp"
     * @param features   niz od 17 double vrednosti (u redosledu iz FeatureExtractor)
     * @param callback   poziva se sa rezultatom ili greškom
     */
    public void sendPrediction(String modelName, double[] features,
                               PredictCallback callback) {
        // Gradi JSON body
        JsonObject body = new JsonObject();
        body.addProperty("model", modelName);

        com.google.gson.JsonArray arr = new com.google.gson.JsonArray();
        for (double f : features) arr.add(f);
        body.add("features", arr);

        String jsonBody = gson.toJson(body);
        Log.d(TAG, "POST " + baseUrl + "/predict  model=" + modelName);

        RequestBody requestBody = RequestBody.create(jsonBody, JSON);
        Request request = new Request.Builder()
                .url(baseUrl + "/predict")
                .post(requestBody)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                String errBody = response.body() != null ? response.body().string() : "(prazan odgovor)";
                Log.e(TAG, "HTTP " + response.code() + ": " + errBody);
                callback.onError("Server greška " + response.code() + ": " + errBody);
                return;
            }

            String responseJson = response.body().string();
            Log.d(TAG, "Odgovor: " + responseJson);

            PredictResult result = gson.fromJson(responseJson, PredictResult.class);
            callback.onSuccess(result);

        } catch (IOException e) {
            Log.e(TAG, "Mrežna greška: " + e.getMessage());
            callback.onError("Mrežna greška: " + e.getMessage()
                    + "\nProveri da li je server pokrenut i da li si na istoj mreži.");
        }
    }

    /**
     * Proverava da li je server dostupan (GET /health).
     * Blokira pozivajući thread.
     *
     * @return true ako server odgovara, false inače
     */
    public boolean checkHealth() {
        Request request = new Request.Builder()
                .url(baseUrl + "/health")
                .get()
                .build();
        try (Response response = client.newCall(request).execute()) {
            return response.isSuccessful();
        } catch (IOException e) {
            Log.w(TAG, "Health check nije uspeo: " + e.getMessage());
            return false;
        }
    }
}
