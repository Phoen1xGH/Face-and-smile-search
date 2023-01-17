package ru.rehtang.service;

import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import ru.rehtang.dto.ImageDto;
import ru.rehtang.feign.UnsplashFeignClient;

import java.io.*;
import java.net.URL;

@Service
@RequiredArgsConstructor
public class NeuralService {
    private final UnsplashFeignClient client;
    @Value("${neuralNetwork.feign.unsplash.apiKey}")
    private String apiKey;


    public ImageDto receivePhoto(String term){
        return client.receivePhoto(term, apiKey);
    }
    public void savePhoto(String term) throws IOException {

        var lol = receivePhoto(term).getUrls().getRegular();
        System.out.println(lol);
        URL url = new URL(lol);
        InputStream in = new BufferedInputStream(url.openStream());
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[1024];
        int n = 0;
        while (-1!=(n=in.read(buf)))
        {
            out.write(buf, 0, n);
        }
        out.close();
        in.close();
        byte[] response = out.toByteArray();
        FileOutputStream fos = new FileOutputStream("C://borrowed_image.jpg");
        fos.write(response);
        fos.close();
    }
}
