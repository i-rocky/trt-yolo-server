#pragma once

#define CPPHTTPLIB_OPENSSL_SUPPORT
#define CPPHTTPLIB_ZLIB_SUPPORT
#define CPPHTTPLIB_BROTLI_SUPPORT
#include "yolov8.h"
#include <httplib.h>

class Handler {
    YoloV8 *yoloV8_;
    httplib::Server server_;
    std::mutex *mutex_;
    std::vector<std::string> mime_types_ = {"image/jpg", "image/jpeg", "image/png", "image/webp"};

    static void split_url(const std::string &url, std::string &host, std::string &path);
    void apply_blur(cv::Mat &image) const;

    static httplib::Server::Handler handleOptionsRequest();
    static std::string base64_decode(const std::string &base64);
    static std::string base64_encode(const std::vector<uchar> &data);
    bool handles(const std::string &content_type) const;
    httplib::Server::Handler handleImageDlRequest() const;
    httplib::Server::Handler handleImageRequest() const;
    void processImage(cv::Mat &image, httplib::Response &res) const;
    static httplib::Server::Handler handleCors();
    static httplib::Server::ExceptionHandler handleException();

public:
    Handler(const char* onnxModelPath, const YoloV8Config &config);
    ~Handler();
    void listen(const std::string &address, int port);
};
