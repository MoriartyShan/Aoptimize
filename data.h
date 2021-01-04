//
// Created by moriarty on 1/4/21.
//

#ifndef ROADCALCULATE_DATA_H
#define ROADCALCULATE_DATA_H
#ifndef RAND_A_DOUBLE
#define RAND_A_DOUBLE(a) \
  ((((rand() & 0xFFFF) / ((scalar)0xFFFF)) - 0.5) * 2 * (a))
#endif
#ifndef DEFAULT_HEIGHT
#define DEFAULT_HEIGHT 4
#endif

scalar inline my_log(const scalar a, const scalar b) {
  return std::log(a) / std::log(b);
}


struct PointsDistancePair {
  static const size_t data_size = 5;
  Pixel2f _point1;
  Pixel2f _point2;
  float _distance;

  PointsDistancePair(){}
  PointsDistancePair(const float *data) {
    _point1.x = data[0];
    _point1.y = data[1];
    _point2.x = data[2];
    _point2.y = data[3];
    _distance = data[4];
  }
  void output(float *data) const {
    data[0] = _point1.x;
    data[1] = _point1.y;
    data[2] = _point2.x;
    data[3] = _point2.y;
    data[4] = _distance;
  }

  void draw_points(cv::Mat &image) {
    cv::circle(image, _point1, 4, CV_RGB(0, 0, 255), -1);
    cv::circle(image, _point2, 4, CV_RGB(0, 0, 255), -1);
  }

  static void write(
      cv::FileStorage &fs,
      const std::string &name,
      const std::vector<PointsDistancePair> &pairs) {
    if (!fs.isOpened()) {
      LOG(FATAL) << "fs is not opened";
    }
    std::vector<std::vector<float>> tmp;
    size_t size = pairs.size();
    tmp.resize(size, std::vector<float>(data_size));
    for (size_t i = 0; i < size; i++) {
      pairs[i].output(tmp[i].data());
    }
//    cv::write(fs, name, tmp);
    fs << name << tmp;
  }

  static void read(
      const cv::FileStorage &fs,
      const std::string &name,
      std::vector<PointsDistancePair> &pairs) {
    if (!fs.isOpened()) {
      LOG(FATAL) << "fs is not opened";
    }
    std::vector<std::vector<float>> tmp;
    auto node = fs[name];
    pairs.clear();
    if (node.isNone()) {
      return;
    }

    cv::read(node, tmp);
    size_t size = tmp.size();
    if (size > 0) {
      pairs.reserve(size);
      for (int i = 0; i < size; i++) {
        pairs.emplace_back(tmp[i].data());
      }
    }
  }
};

struct Params {
  Camera::CameraPtr camera_ptr;
  std::string input_type;
  std::vector<std::vector<Pixel2f>> input_x_lines, input_z_lines;
  std::vector<std::vector<Pixel2f>> x_lines;
  std::vector<std::vector<Pixel2f>> z_lines;
  std::vector<scalar> x_width;
  std::vector<scalar> z_width;
  std::vector<scalar> res;//cam_r_car + height
  std::vector<scalar> best;

  std::vector<PointsDistancePair> points_distance_pairs;

  scalar final_cost;
  scalar cur_min_cost;

  void init(Camera::CameraPtr camera) {
    camera_ptr = camera;
    best = {0, 0, 0, DEFAULT_HEIGHT};
    res = {0, 0, 0, DEFAULT_HEIGHT};
    final_cost = std::numeric_limits<scalar>::max();
    cur_min_cost = final_cost;
  }

  void set_res_zeros() {
    memset(res.data(), 0, sizeof(scalar) * 3);
    res[3] = DEFAULT_HEIGHT;
  }

  void set_res_random() {
    res[0] = RAND_A_DOUBLE(1);
    res[1] = RAND_A_DOUBLE(1);
    res[2] = RAND_A_DOUBLE(1);
    res[3] = DEFAULT_HEIGHT;
  }

  bool isEqualtoBest(const std::vector<scalar>& outside, scalar cost) {
    for (int i = 0; i < outside.size(); i++) {
      if (!DOUBE_EQUAL(outside[i], best[i])) {
        return false;
      }
    }
    return DOUBE_EQUAL(cost, cur_min_cost);
  }

  void update_res(std::vector<double> &res, bool update_inside) {
    if (update_inside) {
      if (res.size() != 4) {
        res.resize(4, 0);
      }
      this->res[0] = res[0];
      this->res[1] = res[1];
      this->res[2] = res[2];
      this->res[3] = res[3];
    } else {
      CHECK(res.size() == 4) << res.size();
      res[0] = this->res[0];
      res[1] = this->res[1];
      res[2] = this->res[2];
      res[3] = this->res[3];
    }
  }
};
class PointsInputOutput {
public:
  const std::string _data_path;
  const std::string _xlines = "xlines";
  const std::string _zlines = "zlines";
  const std::string _xwidth = "xwidth";
  const std::string _zwidth = "zwidth";
  const std::string _start = "start";
  const std::string _points_type = "points_type";
  PointsInputOutput(const std::string &data_path) : _data_path(data_path){
    CHECK(!_data_path.empty()) << "must appoint a --data";
  }

  void write(const std::string &points_file, const Params &params) {
    cv::FileStorage config(points_file, cv::FileStorage::WRITE);
    CHECK(config.isOpened());

    const std::string model = "line";
    config << _points_type << params.input_type;//always output undistorted points
    config << _xlines << params.input_x_lines;
    config << _zlines << params.input_z_lines;
    config << _xwidth << params.x_width << _zwidth << params.z_width;
    config << _start << params.best;
    PointsDistancePair::write(config, "points_distance", params.points_distance_pairs);

    config.release();
  }

  void draw_undist_images(Camera::CameraPtr camera_ptr) {
    std::string origin_image_path = _data_path + "/image.png";
    cv::Mat img = cv::imread(origin_image_path, cv::IMREAD_COLOR);

    if (img.empty()) {
      LOG(ERROR) << "origin image (" << origin_image_path << ") is not exist, try image.bmp";
      origin_image_path = _data_path + "/image.bmp";
      img = cv::imread(origin_image_path, cv::IMREAD_COLOR);
    }
    if (img.empty()) {
      LOG(FATAL) << "origin image (" << origin_image_path << ") is not exist, do normal calibrate";
      return;
    }

    std::string undist_image_path = _data_path + "/undist_image.png";
    cv::Mat undist_image = camera_ptr->UndistortImage(img);
    cv::imwrite(undist_image_path, undist_image);
    LOG(ERROR) << "finish undistort image to [" << undist_image_path << "]";
  }

  void read(Params &params) {
    const std::string points_file = _data_path + "/points.yaml";
    cv::FileStorage config(points_file, cv::FileStorage::READ);
    CHECK(config.isOpened()) << "points file:" << points_file
                             << " load fail, make sure it is exist";
    bool distorted = false;

    auto read_lines = [this, &config](
        const std::string &node_name,
        std::vector<std::vector<Pixel2f>> &lines) {
      cv::FileNode &&node = config[node_name];
      CHECK(!node.isNone()) << "do not contains node: " << _xlines;
      cv::read(node, lines);
    };

    auto undistort_points = [&params](
        const std::vector<std::vector<Pixel2f>> &input,
        std::vector<std::vector<Pixel2f>> &output) {
      const auto camera_ptr = params.camera_ptr;
      const scalar boundary = 10;
      const scalar u_range[2] = { boundary, camera_ptr->ImageWidth() - boundary };
      const scalar v_range[2] = { boundary, camera_ptr->ImageHeight() - boundary };
      LOG(ERROR) << "Undisting points";
      output.resize(input.size());
      for (int i = 0; i < input.size(); i++) {
        params.camera_ptr->UndistortPoints(input[i], output[i]);
        for (int j = 0; j < output[i].size(); j++) {
          if (output[i][j].x > u_range[0] && output[i][j].x < u_range[1]
              && output[i][j].y > v_range[0] && output[i][j].y < v_range[1]) {
            //inside this contour after undistort, a good one
          } else {
            LOG(FATAL) << "Should change point " << input[i][j]
                       << " on line[" << i << "] more inside";
          }
        }
      }
    };

    params.input_type = config[_points_type].string();
    LOG(ERROR) << "type:" << params.input_type;
    if (params.input_type == "Raw") {
      distorted = true;
      LOG(FATAL) << "input point type has to be Undistorted";
    } else if (params.input_type == "Undistorted") {
      distorted = false;
    } else {
      LOG(FATAL) << "points_type = " << params.input_type << ", invalid";
    }
    {
      cv::FileNode &&node = config[_xwidth];
      CHECK(!node.isNone()) << "do not contains node: " << _xwidth;
      cv::read(node, params.x_width);
    }
    {
      cv::FileNode &&node = config[_zwidth];
      CHECK(!node.isNone()) << "do not contains node: " << _zwidth;
      cv::read(node, params.z_width);
    }

    read_lines(_xlines, params.input_x_lines);
    read_lines(_zlines, params.input_z_lines);
    CHECK(params.input_x_lines.size() == (params.x_width.size() + 1)
          || (params.x_width.size() == 0 && params.input_x_lines.size() == 0))
            << "inconsistent x line size:" << params.x_width.size() << ","
            << params.input_x_lines.size();
    CHECK(params.input_z_lines.size() == (params.z_width.size() + 1)
          || (params.z_width.size() == 0 && params.input_z_lines.size() == 0))
            << "inconsistent z line size:" << params.z_width.size() << ","
            << params.input_z_lines.size();

    if (distorted) {
      undistort_points(params.input_x_lines, params.x_lines);
      undistort_points(params.input_z_lines, params.z_lines);
    } else {
      params.x_lines = params.input_x_lines;
      params.z_lines = params.input_z_lines;
    }

    auto nnode = config[_start];
    if (!nnode.isNone()) {
      cv::read(nnode, params.res);
      params.best = params.res;
    }

    PointsDistancePair::read(config, "points_distance", params.points_distance_pairs);

    config.release();


    auto draw_line = [](cv::Mat &img, const std::vector<std::vector<Pixel2f>>& lines) {
      for (int i = 0; i < lines.size(); i++) {
        cv::line(img, lines[i][0], lines[i][lines[i].size() - 1], CV_RGB(255, 0, 0));
      }
    };
    std::string origin_image_path = _data_path + "/image.png";
    cv::Mat img = cv::imread(origin_image_path, cv::IMREAD_COLOR);

    if (img.empty()) {
      LOG(ERROR) << "origin image (" << origin_image_path << ") is not exist, try image.bmp";
      origin_image_path = _data_path + "/image.bmp";
      img = cv::imread(origin_image_path, cv::IMREAD_COLOR);
    }
    if (img.empty()) {
      LOG(ERROR) << "origin image (" << origin_image_path << ") is not exist, do normal calibrate";
      return;
    }
    if (_points_type == "Raw") {
      cv::Mat line_image = img.clone();
      draw_line(line_image, params.input_x_lines);
      draw_line(line_image, params.input_z_lines);
      cv::imwrite(_data_path + "/image_line.png", line_image);
    }

    std::string undist_image_path = _data_path + "/undist_image.png";
    cv::Mat undist_image = cv::imread(undist_image_path, cv::IMREAD_COLOR);
    if (undist_image.empty()) {
      undist_image = params.camera_ptr->UndistortImage(img);
      cv::imwrite(undist_image_path, undist_image);
    }
    CHECK(!undist_image.empty());
    {
      draw_line(undist_image, params.x_lines);
      draw_line(undist_image, params.z_lines);

      for (auto &pair : params.points_distance_pairs) {
        pair.draw_points(undist_image);
      }
      cv::imwrite(_data_path + "/undist_image_line.png", undist_image);

    }

    cv::imwrite("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/undist_undist_2005181158164399.png", img);
  }

  void insert_points(Params &parameters, const int points_number = 80) {
    std::vector<scalar> places;

#define USE_LOG(a) my_log( a, 3)
    places.reserve(points_number);

    for (int i = 0; i < points_number; i++) {
      places.emplace_back(USE_LOG(i + 1));
    }

    auto FillXPoints = [&points_number, &places](std::vector<Pixel2f> &line) {
      if (line.empty()) {
        return;
      }
      std::sort(line.begin(), line.end(), [](const Pixel2f &l, const Pixel2f &r) {
        if (l.y > r.y || (l.y == r.y && l.x < r.x)) return true;
        else return false;
      });

      line.resize(points_number);
      Pixel2f max_dist = (line[1] - line[0]);
      for (int i = 1; i < points_number; i++) {
        line[i] = line[0] + max_dist * places[i] / places.back();
      }

      std::sort(line.begin(), line.end(), [](const Pixel2f &l, const Pixel2f &r) {
        if (l.y > r.y || (l.y == r.y && l.x < r.x)) return true;
        else return false;
      });
      return;
    };

    for (int i = 0; i < parameters.x_lines.size(); i++) {
      FillXPoints(parameters.x_lines[i]);
      LOG(ERROR) << "line " << i << " size = " << parameters.x_lines[i].size();
    }

    //fill in z points
    for (int i = 0; i < parameters.z_lines.size(); i++) {
      LOG(ERROR) << i << ":" << parameters.z_lines[i][0] << "," << parameters.z_lines[i][1] << "," << parameters.z_lines[i].size();
      Pixel2f step = (parameters.z_lines[i][1] - parameters.z_lines[i][0]);
      step /= (points_number - 1);
      parameters.z_lines[i].resize(points_number);
      for (int j = 1; j < points_number; j++) {
        parameters.z_lines[i][j] = parameters.z_lines[i][0] + j * step;
      }
    }
  }
};

#endif //ROADCALCULATE_DATA_H
