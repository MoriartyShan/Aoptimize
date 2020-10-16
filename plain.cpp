#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <gflags/gflags.h>
#include <fstream>
#include "../Common/camera.h"
#include "../Common/utils.h"

DEFINE_int32(rand_seed, -1, "> 1 as seed, 1 is special,"
  " <= 0 for random");
DEFINE_string(data, "", "path to camera.yaml and points.yaml");
DEFINE_bool(has_init, false, "if set true, it start "
  "from the value write in points.yaml");

#define PERFECT_THRESHOLD 0.002
#define GOOD_THRESHOLD
#define RAND_A_DOUBLE(a) \
  ((((rand() & 0xFFFF) / ((double)0xFFFF)) - 0.5) * 2 * (a))

#define DEFAULT_HEIGHT 4
#define STD_DISTANCE 3.8
std::string toString(double *matrix) {
  std::stringstream matrix_string;
  matrix_string.precision(8);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      matrix_string << matrix[i * 3 + j] << ",";
    }
    matrix_string << "\n";
  }
  return matrix_string.str();
}

//only for 3x30.208046,0.0889651,0.00700095,4.64015
template<typename T>
void MatrixMulti(const T *left, const T *right, T* result) {
  const int rown = 3, coln = 3;
  for (int i = 0; i < rown; i++) {
    for (int j = 0; j < coln; j++) {
      int idx = i * coln + j;
      result[idx] = (T)0;
      for (int m = 0; m < coln; m++) {
        result[idx] += left[i * coln + m] * right[m * coln + j];
      }
    }
  }
}

template<typename T>
std::array<T, 2> GetDeviation(const std::vector<std::array<T, 2>> &vectors, const std::array<T, 2>& ave) {
  std::array<T, 2> res = { (T)0 };
  for (auto &v : vectors) {
    T d = v[0] - ave[0];
    res[0] += (d * d);

    d = v[1] - ave[1];
    res[1] += (d * d);
  }
  res[0] /= (T)vectors.size();
  res[1] /= (T)vectors.size();
  return res;
}

template<typename T>
void InsertMatrixToPointer(const cv::Matx33d &m, T *elem) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      elem[i * 3 + j] = (T)m(i, j);
    }
  }
  return;
}

enum RESAULT_LEVEL {
  PERFECT = 0,
  GOOD = 1,
  BAD = 2
};
struct Params {
  double std_distance;
  Camera::CameraPtr camera_ptr;
  std::vector<std::vector<cv::Point2d>> points;
  std::vector<double> res;
  int idx;
  double final_cost;

  double cur_min_cost;
  std::vector<double> best;

  std::vector<int> size;//each image has @size lines
  void set_res_zeros() {
    memset(res.data(), 0, sizeof(double) * 3);
    res[3] = DEFAULT_HEIGHT;
  }

  int get_start() const {
    int start = 0;
    for (int i = 0; i < idx; i++) {
      start += size[i];
    }
    return start;
  }
  int get_end(const int start) const {
    CHECK(idx >= 0) << idx;
    return start + size[idx];
  }

  void set_res_random() {
    res[0] = RAND_A_DOUBLE(1);
    res[1] = RAND_A_DOUBLE(1);
    res[2] = RAND_A_DOUBLE(1);
    res[3] = DEFAULT_HEIGHT;
  }

  bool isEqualtoBest(const std::vector<double>& outside, double cost) {
    for (int i = 0; i < outside.size(); i++) {
      if (!DOUBE_EQUAL(outside[i], best[i])) {
        return false;
      }
    }
    return DOUBE_EQUAL(cost, cur_min_cost);
  }


};

// a line that all points' z value is different
class ZLine {
private:
  std::vector<cv::Point2d> points;
  cv::Matx33d _cameraK;
public:
  static ceres::CostFunction* Create(const cv::Matx33d& cameraK, const std::vector<cv::Point2d> &_points) {
    return (new ceres::AutoDiffCostFunction<ZLine, 1, 4>(
      new ZLine(cameraK, _points)));
  }

  ZLine(const cv::Matx33d& cameraK, const std::vector<cv::Point2d> &_points) {
    _cameraK = cameraK;
    points = _points;
  };

  template <typename T>
  bool operator()(const T* const aim, T* residuals) const {
    std::vector<std::array<T, 2>> psz(points.size());
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    std::array<T, 2> average = { (T)0 };;
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(aim, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    for (int i = 0; i < points.size(); i++) {
      T uv[2] = {(T)points[i].x, (T)points[i].y};
      T p3d[3];
      Camera::GetPoint3d(m_matrix, uv, aim[3], p3d);
      psz[i][1] = p3d[2];
      psz[i][0] = p3d[0];
      average[0] += psz[i][0];
      average[1] += psz[i][1];
    }
    average[0] /= (T)points.size();
    average[1] /= (T)points.size();
    auto dev = GetDeviation(psz, average);
    residuals[0] = ceres::abs(ceres::sqrt(dev[0]) / average[0]);
    return true;
  };
};

class ZLineDistance {
private:
  std::vector<cv::Point2d> _line1;
  std::vector<cv::Point2d> _line2;
  cv::Matx33d _cameraK;
  double _distance;
public:
  static ceres::CostFunction* Create(
      const cv::Matx33d& cameraK,
      const double distance,
      const std::vector<cv::Point2d>& line1,
      const std::vector<cv::Point2d>& line2) {
    return (new ceres::AutoDiffCostFunction<ZLineDistance, 1, 4>(
      new ZLineDistance(cameraK, distance, line1, line2)));
  }

  ZLineDistance(
      const cv::Matx33d& cameraK,
      const double distance,
      const std::vector<cv::Point2d>& line1,
      const std::vector<cv::Point2d>& line2) {
    _cameraK = cameraK;
    _distance = distance;
    _line1 = line1;
    _line2 = line2;
  };

  template <typename T>
  bool operator()(const T* const aim, T* residuals) const {
    std::vector<std::array<T, 2>> line1(_line1.size());
    std::vector<std::array<T, 2>> line2(_line2.size());
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_Matrix(Cam_R_Car);

    ceres::AngleAxisToRotationMatrix(aim, Cam_R_Car_Matrix);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);

    std::array<T, 2> average = { (T)0 };

    for (int i = 0; i < _line1.size(); i++) {
      T uv[2] = { (T)_line1[i].x, (T)_line1[i].y };
      T p3d[3];

      Camera::GetPoint3d(m_matrix, uv, aim[3], p3d);
      line1[i][0] = p3d[0];
      line1[i][1] = p3d[2];
      average[0] += line1[i][0];
    }

    for (int i = 0; i < _line2.size(); i++) {
      T uv[2] = { (T)_line2[i].x, (T)_line2[i].y };
      T p3d[3];
      Camera::GetPoint3d<T>(m_matrix, uv, aim[3], p3d);
      line2[i][0] = p3d[0];
      line2[i][1] = p3d[2];

      average[1] += line2[i][0];
    }

    T max = (T)0, min = (T)100;
    int l1, l2;
    for (int i = 0; i < line1.size(); i++) {
      for (int j = 0; j < line2.size(); j++) {
        T dist = (line1[i][0] - line2[j][0]);
        //LOG(ERROR) << dist;
        if (dist < (T)0) {
          dist = -dist;
        }
        if (max < dist) {
          max = dist;
          l1 = i;
          l2 = j;
        }
        if (min > dist) {
          min = dist;
        }
      }
    }
    max = max - (T)_distance;
    min = min - (T)_distance;
    residuals[0] = max * max + min * min;
    return true;
  };
};

void Ax_B(double a, double b, double c,
  double m, double n, double p) {
  double x = (c / b - p / n) / (a / b - m / n);
  double y = (c / a - (p / m)) / (b / a - (n / m));
  LOG(ERROR) << "res = " << x << "," << y;
}

void write_point(const std::string &points_file,
    const std::vector<std::vector<cv::Point2d>> &points) {
  cv::FileStorage config(points_file, cv::FileStorage::WRITE);
  CHECK(config.isOpened());
  cv::FileNode node;
  config << "image1"
         << "{" << "line1" << points[0] << "line2" << points[1]
         << "line3" << points[2] << "line4" << points[3] << "}";
  config << "image2"
         << "{" << "line1" << points[4] << "line2" << points[5]
         << "line3" << points[6] << "line4" << points[7] << "}";

  config.release();
}

void read_points(const std::string &points_file,
  Params &params) {
  cv::FileStorage config(points_file, cv::FileStorage::READ);
  CHECK(config.isOpened()) << "points file:" << points_file
      << " load fail, make sure it is exist";
  auto &&points = params.points;
  bool distorted = false;
  std::string type = config["points_type"].string();
  if (type == "Raw") {
    distorted = true;
  } else if (type == "Undistorted") {
    distorted = false;
  } else {
    LOG(FATAL) << "points_type = " << type << ", invalid";
  }

  points.clear();
  for (int i = 1; ; i++) {
    std::string name = "image" + std::to_string(i);
    cv::FileNode &&node = config[name];
    if (node.isNone()) {
      break;
    } else {
      int cur_size = points.size();
      MLOG() << "read " << name;
      points.reserve(cur_size + 4);

      for (int i = 0; ; i++) {
        std::string node_name = "line" + std::to_string(i + 1);
        const cv::FileNode &sub_node = node[node_name];
        if (sub_node.isNone()) {
          LOG(ERROR) << "nothing named " << node_name << ", only " << i
                     << " line input";
          params.size.emplace_back(i);
          break;
        }
        LOG(ERROR) << node_name;
        points.emplace_back();
        cv::read(sub_node, points[cur_size + i]);
        LOG(ERROR) << "read sub node: " << node_name;
        std::string pppp;
        for (int j = 0; j < points[cur_size + i].size(); j++) {
          pppp += ("[" + std::to_string(points[cur_size + i][j].x) + ","
                   + std::to_string(points[cur_size + i][j].y) + "]");
        }
        LOG(ERROR) << pppp;
      }
    }
  }
//  exit(0);
  if (distorted) {
    const double boundary = 30;
    const double u_range[2] = { boundary, params.camera_ptr->ImageWidth() - boundary };
    const double v_range[2] = { boundary, params.camera_ptr->ImageHeight() - boundary };
    std::vector<cv::Point2d> tmp;
    LOG(ERROR) << "Undisting points";
    for (int i = 0; i < points.size(); i++) {
      params.camera_ptr->UndistortPoints(points[i], tmp);
      for (int j = 0; j < tmp.size(); j++) {
        if (tmp[j].x > u_range[0] && tmp[j].x < u_range[1]
            && tmp[j].y > v_range[0] && tmp[j].y < v_range[1]) {
          //inside this contour after undistort, a good one
        } else {
          LOG(FATAL) << "Should change point " << points[i][j]
              << " on line[" << i << "] more inside";
        }
      }
      points[i] = tmp;
    }
  }

  auto nnode = config["start"];
  if (!nnode.isNone()) {
    cv::read(nnode, params.res);
  }
  config.release();
#if 0
  MLOG() << "points = " << points.size();
  for (int i = 0; i < points.size(); i++) {
    std::string contain;
    for (int j = 0; j < points[i].size(); j++) {
      contain += ("[" + std::to_string(points[i][j].x) + "," + std::to_string(points[i][j].y) + "]");
    }
    MLOG() << "contain " << contain;
  }
#endif//
}

//make a rotate vector length is inside 2 * Pi
void shrink_to_pi(cv::Vec3d &rvec) {
  double mod = cv::norm(rvec);
  rvec /= mod;
  while (mod > M_PI) {
    mod -= (2 * M_PI);
  }
  rvec *= mod;
  return;
}

void shrink_to_pi(double *vec) {
  cv::Vec3d rvec(vec[0], vec[1], vec[2]);
  shrink_to_pi(rvec);
  vec[0] = rvec(0);
  vec[1] = rvec(1);
  vec[2] = rvec(2);
  return;
}

RESAULT_LEVEL isResultGood(
    const Params* input_params
    ) {
  const double &dist = input_params->std_distance;
  const Camera::CameraPtr camera_ptr = input_params->camera_ptr;
  const std::vector<std::vector<cv::Point2d>> &points = input_params->points;
  const double *res = input_params->res.data();
  const double height_threshold = 0.5;
  const double perfect_threshold = PERFECT_THRESHOLD;
  const double angle_threshold = 0.4330;
  const double tr_threshold = 0.9 * 3;

  if (input_params->res[3] < height_threshold) {
    //judge if height is too small
    MLOG() << "result height too small = "
           << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
           << input_params->final_cost << "," << input_params->cur_min_cost;
    return RESAULT_LEVEL::BAD;
  }

  if (input_params->final_cost < perfect_threshold) {
    MLOG() << "get a perfect result = "
      << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
      << input_params->final_cost << "," << input_params->cur_min_cost;
    return RESAULT_LEVEL::PERFECT;
  }

  double residuals;
  const int start = input_params->get_start();
  const int end = input_params->get_end(start);

  std::vector<ZLine> residual_cal;

  residual_cal.reserve(end - start);
  for (int i = start; i < end; i++) {
    residual_cal.emplace_back(camera_ptr->Intrinsic(), points[i]);
  }

  for (int i = 0; i < residual_cal.size(); i++) {
    residual_cal[i](res, &residuals);
    if (residuals > 1) {
      MLOG() << "bad result because of residual["
             << i << "]" << residuals << "-"
             << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
             << input_params->final_cost << "," << input_params->cur_min_cost;
      return RESAULT_LEVEL::BAD;
    }
  }

  cv::Vec3d rv(res[0], res[1], res[2]);
  cv::Matx33d rotation;
  shrink_to_pi(rv);

  if (cv::norm(rv) > angle_threshold) {
    //(0.25, 0.25, 0.25), whose rotation matrix is 0.9-0.9-0.9 tr
    MLOG() << "too much rotation"
           << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
           << input_params->final_cost << "," << input_params->cur_min_cost;
    return RESAULT_LEVEL::BAD;
  }

  cv::Rodrigues(rv, rotation);

  double tr = cv::trace(rotation);
  if (tr < tr_threshold) {
    MLOG() << "Too small tr " << tr << "\n" << rotation
           << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
           << input_params->final_cost << "," << input_params->cur_min_cost;
    return RESAULT_LEVEL::BAD;
  }

  cv::Matx33d m_Matrix = camera_ptr->Intrinsic() * rotation;
  std::vector<std::vector<std::array<double, 3>>> point3ds(end - start,
      std::vector<std::array<double, 3>>(points[0].size()));
  for (int i = start; i < end; i++) {
    int pos = i - start;
    for (int j = 0; j < points[i].size(); j++) {
      double uv[2] = { points[i][0].x, points[i][0].y };
      Camera::GetPoint3d(m_Matrix.val, uv, res[3], point3ds[pos][j].data());
      if (point3ds[pos][j][2] < 0) {
        MLOG() << "all z value should be positive [" << point3ds[pos][j][0] << ","
               << point3ds[pos][j][1] << "," << point3ds[pos][j][2] << "]"
               << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
               << input_params->final_cost << "," << input_params->cur_min_cost;
        return RESAULT_LEVEL::BAD;
      }
    }

    if (i != start) {
      if (point3ds[pos][0][0] - point3ds[pos - 1][0][0] - dist > 0.5) {
        MLOG() << "two line distance is too much " << pos << "=" << point3ds[pos][0][0]
               << "," << point3ds[pos - 1][0][0] << ","
               << point3ds[pos][0][0] - point3ds[pos - 1][0][0]
               << "\n[" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
               << input_params->final_cost << "," << input_params->cur_min_cost;
        return RESAULT_LEVEL::BAD;
      }
    }
  }
  MLOG() << "get a good start [" << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "],"
         << input_params->final_cost << "," << input_params->cur_min_cost;
  return RESAULT_LEVEL::GOOD;
}

bool Optimize(Params &input_params, ceres::LoggingType log_type = ceres::LoggingType::SILENT) {
  auto &camera_ptr = input_params.camera_ptr;
  auto &points = input_params.points;
  ceres::Problem problem;

  const int start = input_params.get_start();
  const int end = input_params.get_end(start);
  CHECK(end <= points.size()) << "wrong idx";
  double stddist[] = {4.2, 3.57, 3.8};
  for (int i = start; i < end; i++) {
    if (points[i].size() == 0) {
      CHECK(i == end - 1) << i << "," << end;
      break;
    }

    auto cost_func = ZLine::Create(camera_ptr->Intrinsic(), points[i]);
    problem.AddResidualBlock(cost_func, nullptr, input_params.res.data());
    if (i != start) {
      cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(),
                                        stddist[i - start - 1], points[i - 1], points[i]);
      problem.AddResidualBlock(cost_func, nullptr, input_params.res.data());
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  options.logging_type = log_type;
  //options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (log_type != ceres::SILENT) {
    MLOG() << summary.BriefReport();
  }

  if (summary.termination_type == ceres::CONVERGENCE) {
    shrink_to_pi(input_params.res.data());
    input_params.final_cost = summary.final_cost;

    if (input_params.final_cost < input_params.cur_min_cost
        && RESAULT_LEVEL::GOOD >= isResultGood(&input_params)) {
      MLOG() << "final_cost " << input_params.final_cost;
      //only if this is a better result than current
      input_params.cur_min_cost = input_params.final_cost;
      input_params.best = input_params.res;
      return true;
    }
    return false;
  }
  return false;
}

void show_points(const Params &parameters, std::string& result) {
  cv::Vec3d r(parameters.res[0], parameters.res[1], parameters.res[2]);
  cv::Matx33d R;
  cv::Rodrigues(r, R);

  std::stringstream ss;
  ss << "best result = [" << parameters.res[0] << "," << parameters.res[1]
             << "," << parameters.res[2] << "," << parameters.res[3] << "] " << parameters.final_cost << std::endl;

  ss << "rotation = \n" << R << std::endl;
  ss << "car_R_cam = \n" << R.inv() << std::endl;

  cv::Matx33d m_Matrix_ceres = parameters.camera_ptr->Intrinsic() * R;
  result = ss.str();
  ss.str("");
  for (int i = 0; i < parameters.points.size(); i++) {
    ss << "line [" << i << "]\n";
    for (int j = 0; j < parameters.points[i].size(); j++) {
      double uv[2] = { parameters.points[i][j].x, parameters.points[i][j].y };
      double p3d[3];
      Camera::GetPoint3d(m_Matrix_ceres.val, uv, parameters.res[3], p3d);
      ss << "\t[" << j << "]-[" << uv[0] << "," << uv[1] << "]-[" << p3d[0] << "," << p3d[1] << "," << p3d[2] << "]\n";
    }
    result += ss.str();
    ss.str("");
  }
  return;
}

RESAULT_LEVEL optimize_from_zeros(Params &parameter) {
  const int img_number = parameter.size.size();
  for (int i = 0; i < img_number; i++) {
    parameter.set_res_zeros();
    parameter.idx = i;
    Optimize(parameter);
  }
  if (img_number > 1) {
    parameter.set_res_zeros();
    parameter.idx = -1;
    Optimize(parameter);
  }
  parameter.res = parameter.best;
  parameter.final_cost = parameter.cur_min_cost;
  return isResultGood(&parameter);
}

RESAULT_LEVEL optimize_from_set(Params &parameter) {
  const int img_number = parameter.size.size();
  const std::vector<double> init = parameter.res;
  bool has_good = false;
  for (int i = 0; i < img_number; i++) {
    parameter.res = init;
    parameter.idx = i;
    if (Optimize(parameter)) {
      has_good = true;
    }
  }
  if (img_number > 1) {
    parameter.res = init;
    parameter.idx = -1;
    if (Optimize(parameter)) {
      has_good = true;
    }
  }
  if (has_good) {
    parameter.res = parameter.best;
    parameter.final_cost = parameter.cur_min_cost;
    return isResultGood(&parameter);
  }
  return RESAULT_LEVEL::BAD;

}

RESAULT_LEVEL optimize_from_random(Params &parameter) {
  const int img_number = parameter.size.size();
  RESAULT_LEVEL result = RESAULT_LEVEL::BAD;
  while (true) {
    for (int i = 0; i < img_number; i++) {
      parameter.set_res_random();
      parameter.idx = i;
      if (Optimize(parameter)) {
        result = isResultGood(&parameter);
        if (result != BAD) {
          return result;
        }
      }
    }

    if (img_number > 1) {
      parameter.set_res_random();
      parameter.idx = -1;
      if (Optimize(parameter)) {
        result = isResultGood(&parameter);
        if (result != BAD) {
          return result;
        }
      }
    }
  }
}

//return true if it get a perfect result
//return false if not
bool optimize_from_good(Params &parameters) {
  std::vector<double> cur_best;
  double cur_min_cost;
  RESAULT_LEVEL current_result_level = RESAULT_LEVEL::BAD;
  do {
    cur_best = parameters.best;
    cur_min_cost = parameters.cur_min_cost;
    current_result_level = optimize_from_set(parameters);
    if ((current_result_level == RESAULT_LEVEL::BAD)
        || ((current_result_level == RESAULT_LEVEL::GOOD) && parameters.isEqualtoBest(cur_best, cur_min_cost))) {
      //bad result or the same iterate
      MLOG() << "stop this good start";
      return false;
    }
  } while (current_result_level == RESAULT_LEVEL::GOOD);
  return true;
}

double my_log(const double a, const double b) {
  return std::log(a) / std::log(b);
}

void load_points(Params &parameters) {
  const int points_number = 80;
  read_points(FLAGS_data + "/points.yaml", parameters);
  std::vector<double> places;

#define USE_LOG(a) my_log( a, 10)
  places.reserve(points_number);

  for (int i = 0; i < points_number; i++) {
    places.emplace_back(USE_LOG(i + 1));
  }

  auto FillPoints = [&parameters, &points_number, &places](const int id) {
    auto &line = parameters.points[id];
    if (line.empty()) {
      return;
    }
    std::sort(line.begin(), line.end(), [](const cv::Point2d &l, const cv::Point2d &r) {
      if (l.y > r.y || (l.y == r.y && l.x < r.x)) return true;
      else return false;
    });

    line.resize(points_number);
    cv::Point2d max_dist = (line[1] - line[0]);
    for (int i = 1; i < points_number; i++) {
      line[i] = line[0] + max_dist * places[i] / places.back();
    }

    std::sort(line.begin(), line.end(), [](const cv::Point2d &l, const cv::Point2d &r) {
      if (l.y > r.y || (l.y == r.y && l.x < r.x)) return true;
      else return false;
    });
    return;
  };
  cv::Mat img = cv::imread("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1023/undist.png", cv::IMREAD_UNCHANGED);
  for (int i = 0; i < parameters.points.size(); i++) {
    FillPoints(i);
    cv::line(img, parameters.points[i][0], parameters.points[i][points_number - 1], CV_RGB(255, 0, 0));
  }
  cv::imwrite("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1023/undist_undist_2005181158164399.png", img);
}

int main(int argc, char **argv) {
  google::SetVersionString("1.0.0");
  google::SetUsageMessage(std::string(argv[0]) + " [OPTION]");
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  std::string camera_file = FLAGS_data + "/camera.yaml";

  unsigned int rand_seed = FLAGS_rand_seed;
  if (FLAGS_rand_seed <= 0) {
    rand_seed = time(0);
  }
  srand(rand_seed);
  MLOG() << "rand seed = " << rand_seed;

  auto camera_ptr = Camera::create(camera_file);
  Params parameters = {
    STD_DISTANCE,
    camera_ptr,
    std::vector<std::vector<cv::Point2d>>(),
    {0, 0, 0, DEFAULT_HEIGHT},
    0,
    0,
    std::numeric_limits<double>::max(),
    {0, 0, 0, DEFAULT_HEIGHT},
  };
  load_points(parameters);
  RESAULT_LEVEL current_result_level = RESAULT_LEVEL::BAD;
  if (FLAGS_has_init) {
    current_result_level = optimize_from_set(parameters);
  }

  if (current_result_level != RESAULT_LEVEL::PERFECT) {
    //always try zeros if current_result_level is not perfect
    current_result_level = optimize_from_zeros(parameters);
  }

  while (current_result_level != RESAULT_LEVEL::PERFECT) {
    if (current_result_level == RESAULT_LEVEL::GOOD) {
      if (optimize_from_good(parameters)) {
        current_result_level = RESAULT_LEVEL::PERFECT;
        break;
      }
    }
    current_result_level = optimize_from_random(parameters);
  }

  camera_ptr->set_Extrinsic(parameters.res.data());
  camera_ptr->WriteToYaml(FLAGS_data + "/extrinsic.yaml");
  std::string string_result;
  show_points(parameters, string_result);
  for (int i = 0; i < string_result.size(); i+=100) {
    std::cout << string_result.substr(i, 100);
  }

  KEEP_CMD_WINDOW();

  return 0;
}