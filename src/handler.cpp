//
// Created by rocky on 11/14/24.
//
#include <opencv2/imgcodecs.hpp>

#include "handler.h"

Handler::Handler(const char *onnxModelPath, const YoloV8Config &config) {
    yoloV8_ = new YoloV8(onnxModelPath, config);
    mutex_ = new std::mutex();

    server_.Options("/", handleOptionsRequest());
    server_.Get("/", handleImageDlRequest());
    server_.Post("/q", handleImageRequest());
    server_.set_post_routing_handler(handleCors());
    server_.set_exception_handler(handleException());
}

Handler::~Handler() {
    delete yoloV8_;
    delete mutex_;
}

void Handler::split_url(const std::string &url, std::string &host, std::string &path) {
    if (const size_t path_start = url.find('/', url.find("://") + 3); path_start == std::string::npos) {
        host = url;
        path = "/";
    } else {
        host = url.substr(0, path_start);
        path = url.substr(path_start);
    }
}

httplib::Server::Handler Handler::handleOptionsRequest() {
    return [&](const httplib::Request &req, httplib::Response &res) { res.set_content("OK", "text/plain"); };
}

std::string Handler::base64_decode(const std::string &base64) {
    BIO* bio = BIO_new_mem_buf(base64.data(), static_cast<int>(base64.size()));
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);

    std::vector<char> buffer(base64.size() * 3 / 4);
    size_t decoded_len = BIO_read(bio, buffer.data(), static_cast<int>(buffer.size()));
    BIO_free_all(bio);
    return {buffer.data(), decoded_len};
}

std::string Handler::base64_encode(const std::vector<uchar> &data) {
    BIO* bio = BIO_new(BIO_s_mem());
    BIO* b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(b64, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64, bio);

    BIO_write(bio, data.data(), static_cast<int>(data.size()));
    BIO_flush(bio);

    char *buffer = nullptr;
    const long length = BIO_get_mem_data(bio, &buffer);
    std::string encoded(buffer, length);
    BIO_free_all(bio);
    return encoded;
}

bool Handler::handles(const std::string& content_type) const
{
    return std::any_of(mime_types_.begin(), mime_types_.end(), [&](const std::string& type)
    {
        return type == content_type;
    });
}


httplib::Server::Handler Handler::handleImageDlRequest() const {
    return [&](const httplib::Request &req, httplib::Response &res) {
        const auto url = req.get_param_value("q");
        std::cout << "URL: " << url << std::endl;
        std::string host, path;
        split_url(url, host, path);

        httplib::Client client(host);
        auto result = client.Get(path);

        std::cout << "Status: " << result->status << " Content-Type: " << result->get_header_value("Content-Type") << std::endl;
        if (!handles(result->get_header_value("Content-Type")))
        {
            res.set_content(result->body, result->get_header_value("Content-Type"));
            return;
        }

        std::cout << "Handling: " << result->get_header_value("Content-Type") << std::endl;

        const std::vector<uchar> image_data(result->body.begin(), result->body.end());
        cv::Mat image = cv::imdecode(image_data, cv::IMREAD_COLOR);
        cv::imwrite("image.jpg", image);

        if (processImage(image)) {
            std::vector<uchar> output_image_data;
            cv::imencode(".png", image, output_image_data);
            const std::string response = {output_image_data.begin(), output_image_data.end()};
            res.set_content(response, "image/png");
            return;
        }
        res.set_content(result->body, result->get_header_value("Content-Type"));
    };
}

httplib::Server::Handler Handler::handleImageRequest() const {
    return [&](const httplib::Request &req, httplib::Response &res) {
        const std::string base64 = req.body;
        std::string decoded = base64_decode(base64);
        const std::vector<uchar> image_data(decoded.begin(), decoded.end());

        cv::Mat image = cv::imdecode(image_data, cv::IMREAD_COLOR);

        processImage(image);
        std::vector<uchar> output_image_data;
        cv::imencode(".png", image, output_image_data);
        const std::string response = {output_image_data.begin(), output_image_data.end()};
        res.set_content(response, "image/png");
    };
}

bool Handler::processImage(cv::Mat &image) const {
    const auto startTime = std::chrono::steady_clock::now();
    if (image.empty()) {
        std::cerr << "ERROR: image is empty" << std::endl;
        return false;
    }

    if (image.rows < 15 || image.cols < 15) {
        std::cerr << "ERROR: image is too small" << std::endl;
        return false;
    }

    const auto applied = apply_blur(image);

    // end time
    const auto endTime = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsedSeconds = endTime - startTime;
    std::cout << "Time taken: " << elapsedSeconds.count() << " seconds" << std::endl;

    return applied;
}

bool Handler::apply_blur(cv::Mat &image) const {
    mutex_->lock();
    const auto objects = yoloV8_->detectObjects(image);
    const auto masked = YoloV8::drawObjectLabels(image, objects);
    mutex_->unlock();

    return masked;
}

httplib::Server::ExceptionHandler Handler::handleException() {
    return [&](const httplib::Request &req, httplib::Response &res, const std::exception_ptr &ep) {
        try {
            rethrow_exception(ep);
        } catch (const std::exception &e) {
            std::cerr << "Exception: " << e.what() << std::endl;
            // print_stacktrace();
            std::cerr << "Stacktrace:" << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception" << std::endl;
        }
        const auto fmt = "<h1>Error 500</h1><p>%s</p>";
        res.set_content(fmt, "text/html");
        res.status = httplib::StatusCode::InternalServerError_500;
    };
}

void Handler::listen(const std::string &address, const int port) { server_.listen(address, port); }

httplib::Server::Handler Handler::handleCors() {
    return [&](const httplib::Request &req, httplib::Response &res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        res.set_header("Access-Control-Max-Age", "1728000");
    };
}


int main() {
    const YoloV8Config config = {
        .precision = Precision::INT8,
        .calibrationDataDirectory = "/home/rocky/www/yolov8-trt-cpp/images/val2017",
    };

    auto *handler = new Handler("/home/rocky/www/yolov8-trt-cpp/models/yolo11s-seg.onnx", config);

    std::cout << "Listening on port 8555" << std::endl;
    handler->listen("0.0.0.0", 8555);

    delete handler;

    return 0;
}
