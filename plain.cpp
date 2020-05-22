#include <ceres/rotation.h>
#include <ceres/cost_function.h>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <fstream>
#include "../Common/camera.h"
#include "../Common/utils.h"

DEFINE_int32(rand_seed, -1, "> 1 as seed, 1 is special,"
  " <= 0 for random");
DEFINE_string(camera, "", "path to camera.yaml");
DEFINE_string(point_file, "", "path to points.yaml");

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

//only for 3x3
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

// a line that all points' z value is different
class ZLine {
private:
  static int seq;
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
int ZLine::seq = 0;
class ZLineDistance {
private:
  std::vector<cv::Point2d> _line1;
  std::vector<cv::Point2d> _line2;
  cv::Matx33d _cameraK;
  double _distance;
  static int seq;
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
int ZLineDistance::seq = 0;

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
    std::vector<std::vector<cv::Point2d>> &points) {
  MLOG() << "point = " << points_file;
  cv::FileStorage config(points_file, cv::FileStorage::READ);
  CHECK(config.isOpened());
  for (int i = 1; ; i++) {
    std::string name = "image" + std::to_string(i);
    cv::FileNode &node = config[name];
    if (node.isNone()) {
      break;
    } else {
      int cur_size = points.size();
      MLOG() << "read " << name;
      points.resize(cur_size + 4);
      cv::read(node["line1"], points[cur_size]);
      cv::read(node["line2"], points[cur_size + 1]);
      cv::read(node["line3"], points[cur_size + 2]);
      cv::read(node["line4"], points[cur_size + 3]);
    }
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
void shrink_to_2pi(cv::Vec3d &rvec) {
  double mod = cv::norm(rvec);
  const double pi_2 = 2 * M_PI;
  rvec /= mod;
  while (mod > pi_2) {
    mod -= pi_2;
  }
  rvec *= mod;
  return;
}
#if 0
bool isResultGood(
    const double dist,
    const Camera::CameraPtr camera_ptr,
    const std::vector<std::vector<cv::Point2d>>& points,
    const int idx,
    const double * res) {

  if (res[3] < 0.5) {
    //judge if height is too small
    MLOG() << "result height too small = " << res[0]
           << "," << res[1] << "," << res[2] << "," << res[3];
    return false;
  }

  const int base = idx * 4;
  double residuals;
  ZLine residual_cal[4] = {
    ZLine(camera_ptr->Intrinsic(), points[base]),
    ZLine(camera_ptr->Intrinsic(), points[base + 1]),
    ZLine(camera_ptr->Intrinsic(), points[base + 2]),
    ZLine(camera_ptr->Intrinsic(), points[base + 3])
  };

  for (int i = 0; i < 4; i++) {
    residual_cal[i](res, &residuals);
    if (residuals > 0.001) {
      MLOG() << "bad result " << residuals;
      return false;
    }
  }

  cv::Vec3d rv(res[0], res[1], res[2]);
  cv::Matx33d rotation;
  MLOG() << "before " << rv;
  shrink_to_2pi(rv);
  MLOG() << "after " << rv;

  if (cv::norm(rv) > 0.4330) {
    //(0.25, 0.25, 0.25), whose rotation matrix is 0.9-0.9-0.9 tr
    MLOG() << "too much rotation";
    return false;
  }

  cv::Rodrigues(rv, rotation);

  double tr = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);
  if (tr < 0.9 * 3) {
    MLOG() << "Too small tr " << tr << "\n" << rotation;
    return false;
  }

  cv::Matx33d m_Matrix = camera_ptr->Intrinsic() * rotation;
  std::vector<std::vector<double[3]>> point3ds(4,
      std::vector<double[3]>(points[base].size()));
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < points[base + i].size(); j++) {
      double uv[2] = { points[base + i][0].x, points[base + i][0].y };
      Camera::GetPoint3d(m_Matrix.val, uv, res[3], point3ds[i][j]);
      if (point3ds[i][j][2] < 0) {
        MLOG() << "all z value should be positive [" << point3ds[i][j][0] << ","
               << point3ds[i][j][1] << "," << point3ds[i][j][2] << "]";
        return false;
      }
    }

    if (i > 0) {
      if (point3ds[i][0] - point3ds[i - 1][0] - dist > 0.5) {
        MLOG() << "tow line distance is too much " << point3ds[i][0]
               << "," << point3ds[i - 1][0] << "," << point3ds[i][0] - point3ds[i - 1][0];
        return false;
      }
    }
  }


  return true;

}
#endif

int main(int argc, char **argv) {
  google::SetVersionString("1.0.0");
  google::SetUsageMessage(std::string(argv[0]) + " [OPTION]");
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  std::string camera_file = FLAGS_camera;

  unsigned int rand_seed = FLAGS_rand_seed;
  if (FLAGS_rand_seed <= 0) {
    rand_seed = time(0);
  }
  srand(rand_seed);

  auto camera_ptr = Camera::create(camera_file);
  std::vector<std::vector<cv::Point2d>> points, undistort_points;
  const int points_number = 20;
  read_points(FLAGS_point_file, points);

  auto FillPoints = [&points, &points_number](const int id) {
    auto &line = points[id];
    line.resize(points_number);
    cv::Point2d step = (line[1] - line[0]) / (points_number - 1);
    for (int i = 1; i < points_number; i++) {
      line[i] = line[i - 1] + step;
    }
  };
  //cv::Mat img = cv::imread("D:\\Projects\\Documents\\20200518_D\\undistort/undist_2005181158164399.png", cv::IMREAD_UNCHANGED);
  for (int i = 0; i < 8; i++) {
    FillPoints(i);
    //cv::line(img, points[i][0], points[i][total_number - 1], CV_RGB(255, 0, 0));
  }
  //cv::imwrite("D:\\Projects\\Documents\\20200518_D\\undistort/undist_undist_2005181158164399.png", img);

  for (int i = 0; i < 4; i++) {
    std::sort(points[i].begin(), points[i].end(), [](const cv::Point2d &l, const cv::Point2d &r) {
      if (l.y < r.y || (l.y == r.y && l.x < r.x)) return true;
      else return false;
    });

#if 0
    cv::undistortPoints(points[i], undistort_points[i], camera_ptr->Intrinsic(), \
      camera_ptr->Distortion(), cv::noArray(), camera_ptr->Intrinsic());
#endif
  }
  undistort_points = points;
  const double lane_interval = 3.6, stop_cost = 0.001;
  double min_cost = std::numeric_limits<double>::max(), stop_iter_num = 100;
  std::vector<double> best(4);

#define RAND_A_DOUBLE(a) \
  ((((rand() & 0xFFFF) / ((double)0xFFFF)) - 0.5) * 2 * (a))

  std::ofstream start_r("./start.csv");
  const int start_p = 4;
  while (min_cost > stop_cost && stop_iter_num-- >= 0) {
    //std::vector<double> res = { RAND_A_DOUBLE(1),RAND_A_DOUBLE(1),RAND_A_DOUBLE(1), 1.5 };
    std::vector<double> res = { 0.0289033,-0.0291138,-0.00990462,1.58857 };

    start_r << stop_iter_num << "," << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << std::endl;
    ceres::Problem problem;
    MLOG() << "undist p = " << undistort_points[start_p].size() << " " << camera_ptr->Intrinsic();
    auto cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[start_p]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[1+ start_p]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[2+ start_p]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[start_p+3]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    /////////
    cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[start_p], undistort_points[start_p+1]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[start_p+1], undistort_points[start_p+2]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[start_p+2], undistort_points[start_p+3]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    //LOG(ERROR) << "resudual number = " << problem.NumResidualBlocks() << "," << problem.NumResiduals() << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.logging_type = ceres::SILENT;
    //options.max_num_iterations = 1000;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //LOG(ERROR) << "FindRotation:" << summary.BriefReport();
    //LOG(ERROR) << "vector = " << res[0] << "," << res[1] << "," << res[2] << "," << res[3];
    if (summary.final_cost < min_cost) {
      min_cost = summary.final_cost;
      best = res;
      //LOG(WARNING) << "min cost = " << min_cost;
    }

  }
  start_r.close();
  cv::Vec3d r(best[0], best[1], best[2]);
  cv::Matx33d R;
  cv::Rodrigues(r, R);
  LOG(ERROR) << "best result = [" << best[0] << "," << best[1] << "," << best[2] << "," << best[3] << "] " << min_cost << "," << stop_iter_num;
  LOG(ERROR) << "rotation = \n" << R;
  LOG(ERROR) << "car_R_cam = \n" << R.inv();

  cv::Matx33d m_Matrix_ceres = camera_ptr->Intrinsic() * R;
  LOG(ERROR) << "Cam_R_Car = \n" << toString(R.val);
  LOG(ERROR) << "m matrix = \n" << toString(m_Matrix_ceres.val);

  std::ofstream ppppp("./file.csv");

  for (int i = 0; i < 4; i++) {
    std::stringstream ss;
    ss << "line [" << i << "]\n";
    for (int j = 0; j < undistort_points[start_p +i].size(); j++) {
      double uv[2] = { undistort_points[start_p +i][j].x, undistort_points[start_p+i][j].y };
      double p3d[3];
      Camera::GetPoint3d(m_Matrix_ceres.val, uv, best[3], p3d);
      ss << "[" << j << "]" << p3d[0] << "," << p3d[1] << "," << p3d[2] << "]\n";
      ppppp << p3d[0] << "," << p3d[1] << "," << p3d[2] << std::endl;
    }
    LOG(ERROR) << ss.str();
  }
  ppppp.close();

  ZLine lines[4] = {
    ZLine(camera_ptr->Intrinsic(), undistort_points[start_p]),
    ZLine(camera_ptr->Intrinsic(), undistort_points[start_p+1]),
    ZLine(camera_ptr->Intrinsic(), undistort_points[start_p+2]),
    ZLine(camera_ptr->Intrinsic(), undistort_points[start_p+3])
  };
  ZLineDistance zdist[3] = {
    ZLineDistance(camera_ptr->Intrinsic(), lane_interval, undistort_points[start_p], undistort_points[start_p+1]),
    ZLineDistance(camera_ptr->Intrinsic(), lane_interval, undistort_points[start_p+1], undistort_points[start_p+2]),
    ZLineDistance(camera_ptr->Intrinsic(), lane_interval, undistort_points[start_p+2], undistort_points[start_p+3])
  };
  double residuals[2] = {0};
  double aim[4] = { -0.0151731, -0.0314901, -0.0323597,1.54488 };
  for (int i = 0; i < 4; i++) {
    lines[i](best.data(), residuals);
    //lines[i](aim, residuals+1);
    LOG(ERROR) << "zline resudual[" << i << "] = " << residuals[0] << "-" << residuals[1];
  }

  for (int i = 0; i < 3; i++) {
    zdist[i](best.data(), residuals);
    zdist[i](aim, residuals + 1);
    LOG(ERROR) << "ZLineDistance resudual[" << i << "] = " << residuals[0] << "-" << residuals[1];
  }


  KEEP_CMD_WINDOW();

  return 0;
}