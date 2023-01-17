package ru.rehtang;

import javafx.application.Application;
import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.FontPosture;
import javafx.scene.text.FontWeight;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import lombok.RequiredArgsConstructor;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.ConfigurableApplicationContext;
import ru.rehtang.application.CvUtils;
import ru.rehtang.application.CvUtilsFX;
import ru.rehtang.service.NeuralService;

import javax.annotation.PostConstruct;
import java.io.IOException;

@EnableFeignClients
@SpringBootApplication
@RequiredArgsConstructor

public class NeuralNetworkApplication  {
    public final NeuralService service;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    @PostConstruct
    public void test() throws IOException {
        service.savePhoto("face, smile");
    }

    static ConfigurableApplicationContext context;
    public static void main(String[] args) {
        context = SpringApplication.run(NeuralNetworkApplication.class, args);
        Application.launch(Start.class, args);

    }

    public static class Start extends Application {
        private final String IMG_PATH = "C:\\borrowed_image.jpg";
        private final String OPENCV_PATH = "D:\\OpenCVLib\\\\opencv\\\\sources\\\\data\\\\haarcascades\\";

        @Override
        public void start(Stage stage) throws Exception {
            VBox root = new VBox(15.0);
            root.setAlignment(Pos.CENTER);

            Text text = new Text();
            text.setText("Добро пожаловать!");
            text.setFont(Font.font("Sitka Text", FontWeight.BOLD, FontPosture.REGULAR, 36));
            root.getChildren().add(text);

            Button faceSearchButton = new Button("Поиск лиц на изображении");
            faceSearchButton.setOnAction(this::onClickFaceButton);
            faceSearchButton.setPadding(new Insets(20, 39, 20, 39));
            faceSearchButton.setFont(Font.font("Sitka Text", FontWeight.BOLD, FontPosture.REGULAR, 20));
            root.getChildren().add(faceSearchButton);

            Button eyeSearchButton = new Button("Поиск глаз на изображении");
            eyeSearchButton.setPadding(new Insets(20, 35, 20, 35));
            eyeSearchButton.setOnAction(this::onClickEyeButton);
            eyeSearchButton.setFont(Font.font("Sitka Text", FontWeight.BOLD, FontPosture.REGULAR, 20));
            root.getChildren().add(eyeSearchButton);

            Scene scene = new Scene(root, 600.0, 350.0);
            stage.setTitle("OpenCV Application ");
            stage.setScene(scene);
            stage.setOnCloseRequest(event -> {
                Platform.exit();
                context.close();
            });
            stage.show();
        }

        private void onClickFaceButton(ActionEvent e) {
            // Загружаем изображение из файла
            Mat img = Imgcodecs.imread(IMG_PATH);

            if (img.empty()) {
                System.out.println("Не удалось загрузить изображение");
                return;
            }

            faceDetector(img);

        }

        private void faceDetector(Mat img) {
            CascadeClassifier face_detector = new CascadeClassifier();
            String name = "haarcascade_frontalface_alt.xml";

            if (!face_detector.load(OPENCV_PATH + name)) {
                System.out.println("Не удалось загрузить классификатор " + name);
                return;
            }
            MatOfRect faces = new MatOfRect();
            face_detector.detectMultiScale(img, faces);
            for (Rect r : faces.toList()) {
                Imgproc.rectangle(img, new Point(r.x, r.y),
                        new Point(r.x + r.width, r.y + r.height),
                        CvUtils.COLOR_WHITE, 2);
            }
            CvUtilsFX.showImage(img, "Результат");
            img.release();
        }


        private void onClickEyeButton(ActionEvent e) {
            Mat img = Imgcodecs.imread(IMG_PATH);

            if (img.empty()) {
                System.out.println("Не удалось загрузить изображение");
                return;
            }
            eyeDetector(img);

        }

        private void eyeDetector(Mat img) {
            CascadeClassifier face_detector = new CascadeClassifier();
            String name = "haarcascade_frontalface_alt.xml";
            if (!face_detector.load(OPENCV_PATH + name)) {
                System.out.println("Не удалось загрузить классификатор " + name);
                return;
            }
            CascadeClassifier eye_detector = new CascadeClassifier();
            name = "haarcascade_eye.xml";
            if (!eye_detector.load(OPENCV_PATH + name)) {
                System.out.println("Не удалось загрузить классификатор " + name);
                return;
            }
            MatOfRect faces = new MatOfRect();
            face_detector.detectMultiScale(img, faces);
            for (Rect r : faces.toList()) {
                Mat face = img.submat(r);
                MatOfRect eyes = new MatOfRect();
                eye_detector.detectMultiScale(face, eyes);
                for (Rect r2 : eyes.toList()) {
                    Imgproc.rectangle(face, new Point(r2.x, r2.y),
                            new Point(r2.x + r2.width, r2.y + r2.height),
                            CvUtils.COLOR_WHITE, 1);
                }
            }
            CvUtilsFX.showImage(img, "Результат");
            img.release();
        }
    }
}
