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

//distance^2 of point P to line MN
template<typename T>
T GetSquaredDistance(const T *P, const T *M, const T* N) {
  T MN[2] = { N[0] - M[0], N[1] - M[1] };
  T MP[2] = { P[0] - M[0], P[1] - M[1] };
  T cross = MP[0] * MN[1] - MP[1] * MN[0];
  T mod_MN = MN[0] * MN[0] + MN[1] * MN[1];
  cross = cross * cross;
  return cross / mod_MN;
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
    return (new ceres::AutoDiffCostFunction<ZLine, 2, 4>(
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
    }
    
    T dx = psz[0][0] - psz[1][0];
    T dz = psz[0][1] - psz[1][1];
    residuals[0] = (T)100 * dx / dz;
    residuals[1] = dx;
    //residuals[1] = aim[0] * aim[0] + aim[1] * aim[1] + aim[2] * aim[2];
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
    return (new ceres::AutoDiffCostFunction<ZLineDistance, 2, 4>(
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
    }

    for (int i = 0; i < _line2.size(); i++) {
      T uv[2] = { (T)_line2[i].x, (T)_line2[i].y };
      T p3d[3];
      Camera::GetPoint3d<T>(m_matrix, uv, aim[3], p3d);
      line2[i][0] = p3d[0];
      line2[i][1] = p3d[2];
    }

    T vl[2] = { line1[0][0] - line1[1][0],  line1[0][1] - line1[1][1] };
    T vr[2] = { line2[0][0] - line2[1][0],  line2[0][1] - line2[1][1] };
    T cross = vl[0] * vr[1] - vl[1] * vr[0];
    T dot = vl[0] * vr[0] - vl[1] - vr[1];
    
    residuals[0] = (T)100 * ceres::abs(cross / dot);
    residuals[1] = GetSquaredDistance(line1[0].data(), line2[0].data(), line2[1].data());
    residuals[1] += GetSquaredDistance(line1[1].data(), line2[0].data(), line2[1].data());
    residuals[1] += GetSquaredDistance(line2[0].data(), line1[0].data(), line1[1].data());
    residuals[1] += GetSquaredDistance(line2[1].data(), line1[0].data(), line1[1].data());
    residuals[1] -= ((T)4 * (T)(_distance * _distance));
    if (residuals[1] < (T)0) {
      residuals[1] = -residuals[1];
    }
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

int main() {
  std::string camera_file("D:\\Projects\\BoardDetect\\resources\\hardwares\\C.yaml");

  unsigned int rand_seed = FLAGS_rand_seed;
  if (FLAGS_rand_seed <= 0) {
    rand_seed = time(0);
  }
  srand(rand_seed);

  auto camera_ptr = Camera::create(camera_file);
  std::vector<cv::Point2d> points[4], undistort_points[4];

  //calculate points
  const int total_number = 2;
  points[0].resize(total_number);
  points[1].resize(total_number);
  points[2].resize(total_number);
  points[3].resize(total_number);

  points[0][0] = cv::Point2d(65, 749);
  points[0][total_number - 1] = cv::Point2d(935, 501);

  points[1][0] = cv::Point2d(675, 765);
  points[1][total_number - 1] = cv::Point2d(966, 505);

  points[2][0] = cv::Point2d(1410, 755);
  points[2][total_number - 1] = cv::Point2d(1026, 509);

  points[3][0] = cv::Point2d(1811, 666);
  points[3][total_number - 1] = cv::Point2d(1097, 509);

  auto FillPoints = [&points](const int id) {
    auto &line = points[id];
    const int size = line.size() - 1;
    cv::Point2d step = (line[size] - line[0]) / (size);
    LOG(ERROR) << "step = " << step;
    for (int i = 1; i < size; i++) {
      line[i] = line[i - 1] + step;
    }
  };
  //cv::Mat img = cv::imread("D:\\Projects\\Documents\\20200518_D\\undistort/undist_2005181158164399.png", cv::IMREAD_UNCHANGED);
  for (int i = 0; i < 4; i++) {
    FillPoints(i);
    //cv::line(img, points[i][0], points[i][total_number - 1], CV_RGB(255, 0, 0));
  }
  //cv::imwrite("D:\\Projects\\Documents\\20200518_D\\undistort/undist_undist_2005181158164399.png", img);

  for (int i = 0; i < 4; i++) {
    std::sort(points[i].begin(), points[i].end(), [](const cv::Point2d &l, const cv::Point2d &r) {
      if (l.y < r.y || (l.y == r.y && l.x < r.x)) return true;
      else return false;
    });
    undistort_points[i] = points[i];
#if 0
    cv::undistortPoints(points[i], undistort_points[i], camera_ptr->Intrinsic(), \
      camera_ptr->Distortion(), cv::noArray(), camera_ptr->Intrinsic());
#endif
  }
  const double lane_interval = 3.6, stop_cost = 0.001;
  double min_cost = std::numeric_limits<double>::max(), stop_iter_num = 1;
  std::vector<double> best(4);

#define RAND_A_DOUBLE(a) \
  ((((rand() & 0xFFFF) / ((double)0xFFFF)) - 0.5) * 2 * (a))

  std::ofstream start_r("./start.csv");
  
  while (min_cost > stop_cost && stop_iter_num-- >= 0) {
    /*std::vector<double> res = { RAND_A_DOUBLE(1),RAND_A_DOUBLE(1),RAND_A_DOUBLE(1), 1.5 };*/
    std::vector<double> res = { 0,0,0, 1.5 };
    start_r << stop_iter_num << "," << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << std::endl;
    ceres::Problem problem;

    auto cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[0]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[1]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[2]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLine::Create(camera_ptr->Intrinsic(), undistort_points[3]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    /////////
    cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[0], undistort_points[1]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[1], undistort_points[2]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());

    cost_func = ZLineDistance::Create(camera_ptr->Intrinsic(), lane_interval, undistort_points[2], undistort_points[3]);
    problem.AddResidualBlock(cost_func, nullptr, res.data());
    //problem.SetParameterLowerBound(res.data(), 3, 1);
    //LOG(ERROR) << "resudual number = " << problem.NumResidualBlocks() << "," << problem.NumResiduals() << std::endl;

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100000;
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

  for (int i = 0; i < 4; i++) {
    std::stringstream ss;
    ss << "line [" << i << "]\n";
    for (int j = 0; j < undistort_points[i].size(); j++) {
      double uv[2] = { undistort_points[i][j].x, undistort_points[i][j].y };
      double p3d[3];
      Camera::GetPoint3d(m_Matrix_ceres.val, uv, best[3], p3d);
      ss << "[" << j << "]" << p3d[0] << "," << p3d[1] << "," << p3d[2] << "]\n";
    }
    LOG(ERROR) << ss.str();
  }

  ZLine lines[4] = {
    ZLine(camera_ptr->Intrinsic(), undistort_points[0]),
    ZLine(camera_ptr->Intrinsic(), undistort_points[1]),
    ZLine(camera_ptr->Intrinsic(), undistort_points[2]),
    ZLine(camera_ptr->Intrinsic(), undistort_points[3])
  };
  ZLineDistance zdist[3] = {
    ZLineDistance(camera_ptr->Intrinsic(), lane_interval, undistort_points[0], undistort_points[1]),
    ZLineDistance(camera_ptr->Intrinsic(), lane_interval, undistort_points[1], undistort_points[2]),
    ZLineDistance(camera_ptr->Intrinsic(), lane_interval, undistort_points[2], undistort_points[3])
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
    //zdist[i](aim, residuals + 1);
    LOG(ERROR) << "ZLineDistance resudual[" << i << "] = " << residuals[0] << "-" << residuals[1];
  }


  KEEP_CMD_WINDOW();

  return 0;
}