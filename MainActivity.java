package com.example.shapecounterapp;

import  android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.*;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private PreviewView previewView;
    private TextView tvCount;
    private Button btnScreenshot;
    private ExecutorService cameraExecutor;
    private Bitmap latestBitmap = null;

    static {
        OpenCVLoader.initDebug();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        previewView = findViewById(R.id.previewView);
        tvCount = findViewById(R.id.tvCount);
        btnScreenshot = findViewById(R.id.btnScreenshot);

        btnScreenshot.setOnClickListener(v -> {
            if (latestBitmap != null) {
                saveToGallery(latestBitmap);
            } else {
                Toast.makeText(this, "Nema slike", Toast.LENGTH_SHORT).show();
            }
        });

        cameraExecutor = Executors.newSingleThreadExecutor();

        if (allPermissionsGranted()) startCamera();
        else ActivityCompat.requestPermissions(this,
                new String[]{Manifest.permission.CAMERA}, 10);
    }

    // kamera
    // definisanje metode koja pokrece kameru
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(this); //trazi CameraX sistem (kameru) od Androida, ali asinhrono (ne dobija odmah rezultat)

        future.addListener(() -> { // dodaje “callback” koji se izvrsava kada kamera postane spremna
            try {
                ProcessCameraProvider provider = future.get(); // uzimа stvarni CameraProvider objekat (kamera je sada spremna)
                bindCamera(provider); // Poziva metodu koja povezuje kameru sa ekranom i analizom
            } catch (Exception e) { // Ako kamera ne radi ulazi u ovaj blok
                Toast.makeText(this, "Camera error", Toast.LENGTH_SHORT).show(); // prikazuje poruku “Camera error” korisniku
            }
        }, ContextCompat.getMainExecutor(this));
    }

    // metoda koja povezuje kameru sa prikazom i analizom
    private void bindCamera(ProcessCameraProvider provider) {

        // kreira objekat za prikaz live slike kamere na ekranu
        Preview preview = new Preview.Builder().build();

        // kreira objekat koji omogućava obradu svakog frame-a iz kamere
        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST) // koristi samo najnoviji frame,ignore stare da ne kasni
                .build();

        // svaki frame salje u analyzeImage() funkciju na obradu
        analysis.setAnalyzer(cameraExecutor, this::analyzeImage);

        // kreira objekat koji bira kameru (prednja ili zadnja
        CameraSelector selector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK) // biramo zadnju kameru telefona
                .build();

        // povezuje kameru sa UI ekranom (PreviewView)
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

// uklanja sve prethodne veze kamere (reset stanje)
        provider.unbindAll();
        // pokrece kameru i povezuje activity,preview, analizu slike
        provider.bindToLifecycle(this, selector, preview, analysis);
    }

    // analiza slike
    // metoda koja prima jedan frame iz kamere (sliku u realnom vremenu)
    private void analyzeImage(@NonNull ImageProxy imageProxy) {

        // pretvara sliku iz CameraX formata u OpenCV Mat (slika za obradu)
        Mat mat = imageProxyToMat(imageProxy);

        if (mat != null) {

            // salje sliku u funkciju koja pronalazi oblike i broji ih
            int[] result = detectShapes(mat);

            // Kreira praznu Android sliku (Bitmap) iste velicine kao Mat
            // cols() = sirina, rows() = visina, ARGB_8888 = kvalitet slike (full color)
            latestBitmap = Bitmap.createBitmap(
                    mat.cols(),
                    mat.rows(),
                    Bitmap.Config.ARGB_8888
            );

            //Kopira OpenCV sliku u Android Bitmap
            Utils.matToBitmap(mat, latestBitmap);

            // pokrece UI update (jer background thread ne smije mijenjati UI)
            // ispisuje krugove, kvadrate i trouglove
            runOnUiThread(() -> tvCount.setText(
                    "Krugovi: " + result[0] +
                            " | Kvadrati: " + result[1] +
                            " | Trouglovi: " + result[2] +
                            "\nUKUPNO: " + (result[0] + result[1] + result[2])
            ));

            // delete Mat iz memorije (OpenCV cleanup) oslobadja memoriju da app ne puca
            mat.release();
        }

        imageProxy.close();
    }

    // analizira sliku i broji krugove, kvadrate i trouglove pomocu OpenCV kontura
    //Metoda prima sliku u OpenCV formatu (Mat)
    private int[] detectShapes(Mat mat) {

        // Kreira novu praznu sliku za grayscale
        Mat gray = new Mat();
        // Pretvara sliku iz boje u crno-bijelu zboglakse detekcije i manje podataka
        Imgproc.cvtColor(mat, gray, Imgproc.COLOR_BGR2GRAY);
        // blur sliku jer uklanja sum, poboljsava detekiju kontura
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);

        //nova slika za binarni rezultat
        Mat thresh = new Mat();
        //Pretvara sliku u crno-bijelu (0 ili 255)
        //pikseli > 120 - bijelo/crno (zavisi od INV)
        Imgproc.threshold(gray, thresh, 120, 255, Imgproc.THRESH_BINARY_INV);

        // Lista svih pronadjenhi oblika
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();

        // glavna funkcija koja pronalazi oblike na slici
        Imgproc.findContours(
                thresh,
                contours,
                hierarchy,
                Imgproc.RETR_EXTERNAL,
                Imgproc.CHAIN_APPROX_SIMPLE
        );

        // varijable za brojanje oblika
        int circles = 0, squares = 0, triangles = 0;

        // petlja prolazi kroz svaki pronađeni oblik
        for (MatOfPoint c : contours) {

            // racuna size oblika
            double area = Imgproc.contourArea(c);
            // ignorise male objkete, sum
            if (area < 500) continue;

            // pretvara konturu u float verziju (preciznija analiza)
            MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
            MatOfPoint2f approx = new MatOfPoint2f();

            double epsilon = 0.02 * Imgproc.arcLength(c2f, true);
            // pojednostavljuje oblik u manje tacaka
            Imgproc.approxPolyDP(c2f, approx, epsilon, true);

            // broj uglova (stranica)
            int vertices = (int) approx.total();

            if (vertices == 3) triangles++;
            else if (vertices == 4) squares++;
            else if (vertices > 5) circles++;

            c.release();
            c2f.release();
            approx.release();
        }

        gray.release();
        thresh.release();
        hierarchy.release();

        // vraca broj blika
        return new int[]{circles, squares, triangles};
    }

    // ---------------- IMAGE → MAT ----------------

    private Mat imageProxyToMat(ImageProxy image) {

        ImageProxy.PlaneProxy[] planes = image.getPlanes();

        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        byte[] nv21 = new byte[ySize + uSize + vSize];

        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        Mat yuv = new Mat(image.getHeight() + image.getHeight() / 2,
                image.getWidth(),
                CvType.CV_8UC1);

        yuv.put(0, 0, nv21);

        Mat rgb = new Mat();
        Imgproc.cvtColor(yuv, rgb, Imgproc.COLOR_YUV2BGR_NV21);

        yuv.release();

        return rgb;
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (allPermissionsGranted()) startCamera();
        else {
            Toast.makeText(this, "Camera needed", Toast.LENGTH_SHORT).show();
            finish();
        }
    }

    // ---------------- SAVE IMAGE ----------------

    private void saveToGallery(Bitmap bitmap) {

        try {
            String name = "Shape_" + System.currentTimeMillis() + ".png";

            android.content.ContentValues values = new android.content.ContentValues();
            values.put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, name);
            values.put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/png");
            values.put(android.provider.MediaStore.Images.Media.RELATIVE_PATH,
                    "Pictures/ShapeCounter");

            android.net.Uri uri = getContentResolver().insert(
                    android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    values
            );

            if (uri != null) {
                OutputStream out = getContentResolver().openOutputStream(uri);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                out.close();

                Toast.makeText(this, "Spremljeno", Toast.LENGTH_SHORT).show();
            }

        } catch (Exception e) {
            Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        cameraExecutor.shutdown();
        super.onDestroy();
    }
}