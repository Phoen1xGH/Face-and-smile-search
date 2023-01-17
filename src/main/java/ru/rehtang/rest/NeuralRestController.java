package ru.rehtang.rest;

import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import ru.rehtang.dto.ImageDto;
import ru.rehtang.service.NeuralService;

import java.io.IOException;

@RestController
@RequiredArgsConstructor
public class NeuralRestController {
    private final NeuralService service;

    @GetMapping("/getPhoto")
    public ImageDto receivePhoto(
            @RequestParam String term
    ){
        System.out.println(service.receivePhoto(term));
        return  service.receivePhoto(term);

    }

    @GetMapping("/savePhoto")
    public void savePhoto(@RequestParam String term) throws IOException {
        service.savePhoto(term);
    }
}
