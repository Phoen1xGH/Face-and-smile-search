package ru.rehtang.feign;

import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import ru.rehtang.dto.ImageDto;

@FeignClient(name = "unsplashFeignClient", url = "${neuralNetwork.feign.unsplash.url}")
public interface UnsplashFeignClient {
    @GetMapping("/random")
    ImageDto receivePhoto(
            @RequestParam("query") String term,
            @RequestParam("client_id")String key
            );
}
