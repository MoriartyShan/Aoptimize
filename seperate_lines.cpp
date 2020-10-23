//
// Created by moriarty on 10/17/20.
//
#include <stdlib.h>
#include "PointAtAxisLineCost.h"
#include "PointsAxisDistance.h"
#include "../Common/camera.h"
#include "ValueDiffCost.h"

void read_points(
    const Camera::CameraPtr &camera_ptr,
    const std::string &points_file,
    std::vector<std::vector<cv::Point2d>> &lines) {
  cv::FileStorage config(points_file, cv::FileStorage::READ);
  CHECK(config.isOpened()) << "points file:" << points_file
                           << " load fail, make sure it is exist";
  bool distorted = false;
  std::string type = config["points_type"].string();
  if (type == "Raw") {
    distorted = true;
  } else if (type == "Undistorted") {
    distorted = false;
  } else {
    LOG(FATAL) << "points_type = " << type << ", invalid";
  }

  lines.clear();
  for (int i = 1; ; i++) {
    std::string name = "image" + std::to_string(i);
    cv::FileNode &&node = config[name];
    if (node.isNone()) {
      break;
    } else {
      int cur_size = lines.size();
      MLOG() << "read " << name;
      lines.reserve(cur_size + 4);

      for (int i = 0; ; i++) {
        std::string node_name = "line" + std::to_string(i + 1);
        const cv::FileNode &sub_node = node[node_name];
        if (sub_node.isNone()) {
          LOG(ERROR) << "nothing named " << node_name << ", only " << i
                     << " line input";
          break;
        }
        LOG(ERROR) << node_name;
        lines.emplace_back();
        cv::read(sub_node, lines[cur_size + i]);
        LOG(ERROR) << "read sub node: " << node_name;
        std::string pppp;
        for (int j = 0; j < lines[cur_size + i].size(); j++) {
          pppp += ("[" + std::to_string(lines[cur_size + i][j].x) + ","
                   + std::to_string(lines[cur_size + i][j].y) + "]");
        }
        LOG(ERROR) << pppp;
      }
    }
  }
//  exit(0);
//  if (distorted) {
//    const double boundary = 30;
//    const double u_range[2] = { boundary, params.camera_ptr->ImageWidth() - boundary };
//    const double v_range[2] = { boundary, params.camera_ptr->ImageHeight() - boundary };
//    std::vector<cv::Point2d> tmp;
//    LOG(ERROR) << "Undisting points";
//    for (int i = 0; i < points.size(); i++) {
//      params.camera_ptr->UndistortPoints(points[i], tmp);
//      for (int j = 0; j < tmp.size(); j++) {
//        if (tmp[j].x > u_range[0] && tmp[j].x < u_range[1]
//            && tmp[j].y > v_range[0] && tmp[j].y < v_range[1]) {
//          //inside this contour after undistort, a good one
//        } else {
//          LOG(FATAL) << "Should change point " << points[i][j]
//                     << " on line[" << i << "] more inside";
//        }
//      }
//      points[i] = tmp;
//    }
//  }

//  auto nnode = config["start"];
//  if (!nnode.isNone()) {
//    cv::read(nnode, params.res);
//  }
  config.release();

}
double my_log(const double a, const double b) {
  return std::log(a) / std::log(b);
}

#define USE_LOG(a) my_log( a, 10)
void load_points(const Camera::CameraPtr &camera_ptr,
                 const std::string &points_file,
                 std::vector<std::vector<cv::Point2d>> &lines) {
  const int points_number = 80;
  read_points(camera_ptr, points_file, lines);
  std::vector<double> places;

#define USE_LOG(a) my_log( a, 10)
  places.reserve(points_number);

  for (int i = 0; i < points_number; i++) {
    places.emplace_back(USE_LOG(i + 1));
  }

  auto FillPoints = [&lines, &points_number, &places](const int id) {
    auto &line = lines[id];
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
  cv::Mat img = cv::imread("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/undist.png", cv::IMREAD_UNCHANGED);
  for (int i = 0; i < lines.size(); i++) {
    FillPoints(i);
    cv::line(img, lines[i][0], lines[i][points_number - 1], CV_RGB(255, 0, 0));
  }
  cv::imwrite("/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/undist_undist_2005181158164399.png", img);
}


void test_a() {
  std::string camera_path = "/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/camera.yaml";
  std::string points_path = "/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1022/points.yaml";
  auto camera_ptr = Camera::create(camera_path);
  const cv::Matx33d &cameraK = camera_ptr->Intrinsic();

  std::vector<std::vector<cv::Point2d>> lines;
  load_points(camera_ptr, points_path,lines);

  ceres::Problem problem;

  cv::Vec3d cam_r_car(0.208182,0.0874568,0.0143281);
  double height = 4.80547;
  double positions[5] = {-17.7961, -14.0, -9.93168, -5.87605, -1.94166};
  //-17.7961, -14.0, -9.93168, -5.87605, -1.94166
  //0.208182,0.0874568,0.0143281,4.80547
  std::vector<PointAtAxisLine> points_at_line_cost;
  std::vector<ValueDiff> values_diff_cost;
  points_at_line_cost.reserve(80 * 5);
  values_diff_cost.reserve(5);
  for (int i = 0; i < lines.size(); i++) {
    for (int j = 0; j < lines[i].size(); j++) {
      auto cost_func = PointAtAxisLine::Create(lines[i][j], cameraK, 0);
      problem.AddResidualBlock(cost_func, nullptr, cam_r_car.val, &height, positions + i);
      points_at_line_cost.emplace_back(lines[i][j], cameraK, 0);

      if (j > 0) {
        cost_func = PointAxisDistance::Create(lines[i][j], lines[i][0], cameraK, 0, 2);
        problem.AddResidualBlock(cost_func, nullptr, cam_r_car.val, &height);
      }


      if (i != 0) {
        const double distance = positions[i] - positions[i - 1];
        for (int k = 0; k < lines[i - 1].size(); k++) {
          auto cost_func_2 = PointAxisDistance::Create(lines[i][j], lines[i- 1][k], cameraK, distance, 0);
          problem.AddResidualBlock(cost_func_2, nullptr, cam_r_car.val, &height);
        }
      }
    }
  }

  for (int i = 1; i < 5; i++) {
    auto cost_func = ValueDiff::Create(positions[i] - positions[i - 1]);
    problem.AddResidualBlock(cost_func, nullptr, positions + i, positions + i - 1);
    values_diff_cost.emplace_back(positions[i] - positions[i - 1]);
//    problem.SetParameterBlockConstant(positions + i);
//    if (i == 1) {
//      problem.SetParameterBlockConstant(positions);
//    }
  }

//  std::vector<std::vector<cv::Point2d>> z_lines(
//      {{cv::Point2d(814, 326), cv::Point2d(344, 300)},
//       {cv::Point2d(520, 167), cv::Point2d(346, 170)}});
//  double zs[2];
//  std::vector<PointAtAxisLine> zline_cost;
//  zline_cost.reserve(160);
//  {
//    cv::Point2d step;
//    double residuals[2];
//    const int insert_num = 80;
//    for (int i = 0; i < z_lines.size(); i++) {
//      step = (z_lines[i][1] - z_lines[i][0])/insert_num;
//      z_lines[i].resize(insert_num);
//
//      auto cost_func = PointAtAxisLine::Create(z_lines[i][0], cameraK, 1);
//      problem.AddResidualBlock(cost_func, nullptr, cam_r_car.val, &height, zs + i);
//      zline_cost.emplace_back(z_lines[i][0], cameraK, 1);
//      zline_cost.back()(cam_r_car.val, &height, zs + i, residuals);
//      LOG(ERROR) << zline_cost.size() - 1 << "," << i << "," << 0 << " init:" << residuals[0] << "," << residuals[1];
//
//      for (int j = 1; j < insert_num; j++) {
//        z_lines[i][j] = z_lines[i][j - 1] + step;
//
//        cost_func = PointAtAxisLine::Create(z_lines[i][j], cameraK, 1);
//        problem.AddResidualBlock(cost_func, nullptr, cam_r_car.val, &height, zs + i);
//        zline_cost.emplace_back(z_lines[i][j], cameraK, 1);
//        zline_cost.back()(cam_r_car.val, &height, zs + i, residuals);
//        LOG(ERROR) << zline_cost.size() - 1 << "," << i << "," << j << " init:" << residuals[0] << "," << residuals[1];
//      }
//    }
//  }

//  problem.SetParameterBlockConstant(&height);
  problem.SetParameterLowerBound(&height, 0, 4);
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
//  options.logging_type = log_type;
  //options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  MLOG() << summary.BriefReport();

  cv::Matx33d cam_R_car;
  cv::Rodrigues(cam_r_car, cam_R_car);
  cv::Matx33d car_R_cam = cam_R_car.t();

  LOG(ERROR) << cam_r_car << height;
  LOG(ERROR) << std::endl << cam_R_car;

  camera_ptr->set_Car_R_Cam(car_R_cam);
  camera_ptr->set_camera_height(height);
  {
    double residuals[2];
    for (int i = 0; i < lines.size(); i++) {
      LOG(ERROR) << "line:" << positions[i];
      for (int j = 0; j < lines[i].size(); j++) {
        auto p = camera_ptr->GetPoint3d(lines[i][j]);
        points_at_line_cost[i * 80 + j](cam_r_car.val, &height, positions + i, residuals);
        LOG(ERROR) << p << lines[i][j] << "," << residuals[0] << "," << residuals[1] << std::endl;
      }
    }
  }


//  LOG(ERROR) << "\n\n\n";
//  {
//    double residuals[2];
//    for (int i = 0; i < z_lines.size(); i++) {
//      LOG(ERROR) << "zline:" << zs[i];
//      for (int j = 0; j < z_lines[i].size(); j++) {
////        LOG(ERROR) << z_lines[i][j] << i << "," << j << "," << i * 80 + j << "," << z_lines[0].size();
//        auto p = camera_ptr->GetPoint3d(z_lines[i][j]);
//        zline_cost[i * 80 + j](cam_r_car.val, &height, zs + i, residuals);
//        LOG(ERROR) << p << z_lines[i][j] << ", " << residuals[0] << "," << residuals[1];
//      }
//    }
//  }
  {
    double residuals[2];
    for (int i = 1; i < 5; i++) {
      values_diff_cost[i - 1](positions + i, positions + i - 1, residuals);
      LOG(ERROR) << "cost " << i << " = " << residuals[0];
    }
  }
}


class XCalculateParam{
private:
  const cv::Matx33d _cameraK;
  const cv::Point2d _point1;
  const cv::Point2d _point2;
public:
  static ceres::CostFunction* Create(
      const cv::Matx33d& cameraK,
      const cv::Point2d &point1,
      const cv::Point2d &point2) {
    return (new ceres::AutoDiffCostFunction<XCalculateParam, 8, 3, 2, 2>(
        new XCalculateParam(cameraK, point1, point2)));
  }

  XCalculateParam(
      const cv::Matx33d& cameraK,
      const cv::Point2d &point1,
      const cv::Point2d &point2) :
      _cameraK(cameraK), _point1(point1), _point2(point1){};

  template <typename T>
  bool operator()(
      const T* const Cam_r_Car,
      const T* const zm1,
      const T* const zm2,
      T* residuals) const {
    T Cam_R_Car[9], cameraK[9], m_matrix[9];
    ceres::MatrixAdapter<T, 3, 1> m_Matrix(m_matrix), Cam_R_Car_M(Cam_R_Car);
//    LOG(ERROR) << "output residual:" << residuals[0] << residuals[1] << residuals[2];
    ceres::AngleAxisToRotationMatrix(Cam_r_Car, Cam_R_Car_M);
    InsertMatrixToPointer(_cameraK, cameraK);
    MatrixMulti(cameraK, Cam_R_Car, m_matrix);
//    LOG(ERROR) << "output residual:" << residuals[0] << residuals[1] << residuals[2];
    T zd = (zm1[0] - zm2[0]);
    residuals[0] = m_Matrix(0, 2) * zd - (zm1[1] * (T)_point1.x - zm2[1] * (T)_point2.x);
    residuals[1] = m_Matrix(1, 2) * zd - (zm1[1] * (T)_point1.y - zm2[1] * (T)_point2.y);
    residuals[2] = m_Matrix(2, 2) * zd - (zm1[1] - zm2[1]);

    if (zm2[1] < (T)0) {
      residuals[3] = zm2[1];
    } else {
      residuals[3] = (T)0;
    }
    if (zm1[1] < (T)0) {
      residuals[4] = zm1[1];
    } else {
      residuals[4] = (T)0;
    }

    if (zm2[0] < (T)0) {
      residuals[5] = zm2[0];
    } else {
      residuals[5] = (T)0;
    }
    if (zm1[0] < (T)0) {
      residuals[6] = zm1[0];
    } else {
      residuals[6] = (T)0;
    }
    T xd = residuals[0] / m_Matrix(0, 0);
    if (xd != (T)0) {
      residuals[7] = zd / xd;
    } else {
      residuals[7] = (T)0;
    }

//    LOG(ERROR) << zd;
//    LOG(ERROR) << "output residual:" << residuals[0]  << "," << residuals[1] << "," << residuals[2];
    return true;
  };
};

double *ptr = nullptr;
void test_b() {
  std::string path = "/home/moriarty/WindowsD/Projects/WDataSets/HKvisionCalib/extrinsics/1023/";
  std::string camera_path = path + "camera.yaml";
  std::string points_path = path + "points.yaml";
  auto camera_ptr = Camera::create(camera_path);
  const cv::Matx33d &cameraK = camera_ptr->Intrinsic();

  std::vector<std::vector<cv::Point2d>> lines;
  load_points(camera_ptr, points_path,lines);

  ceres::Problem problem;
  std::vector<std::vector<double>> zms;
  cv::Vec3d r;

  std::vector<double> initv(2);
  initv[0] = 1;
  initv[1] = 1;

  LOG(ERROR) << "init v : " << (void *)&initv[0];
  std::vector<XCalculateParam> cost_functions;
  for (int i = 0; i < lines.size(); i++) {
    const int start_idx = zms.size();
    zms.reserve(zms.size() + lines[i].size());
    zms.emplace_back(initv);

    double *left_ptr = zms[start_idx].data();
    LOG(ERROR) << start_idx << "," << i << "," << 0 << "," << (void *)r.val<< "," << (void *)&zms.back()[0] << "," << left_ptr;
    for (int j = 1; j < lines[i].size(); j++) {
      auto cost_func = XCalculateParam::Create(cameraK, lines[i][0], lines[i][j]);
      zms.emplace_back(std::vector<double>({rand() % 10, rand() % 10}));
      cost_functions.emplace_back(cameraK, lines[i][0], lines[i][j]);
      if (i == 0 && (j == 15)) {
        ptr = zms[start_idx + j].data();
        LOG(ERROR) << "ptr = " << (void *)ptr << "," <<ptr[0] << "," << ptr[1];
//        LOG(ERROR) << zms[start_idx + j][0] << "," << zms[start_idx + j][1] << "," << ptr[0] << "," << ptr[1];
//        LOG(ERROR) << "ptr = " << (void *)ptr << "," <<ptr[0] << "," << ptr[1];
      }

      double* right_ptr = zms[start_idx + j].data();
      LOG(ERROR) << start_idx << "," << i << "," << j << "," << (void*)ptr << "," << (void *)r.val<< "," << (void *)left_ptr << "," << (void *)right_ptr << "," << start_idx + j;
      problem.AddResidualBlock(cost_func, nullptr, r.val, left_ptr, right_ptr);
    }
  }
  LOG(ERROR) << "ptr = " << (void *)ptr << "," <<ptr[0] << "," << ptr[1];
  LOG(ERROR) << "add zm " << zms.size();
  LOG(ERROR) << "num = " << problem.NumResidualBlocks() << "," << problem.NumResiduals();
//  problem.SetParameterLowerBound(&height, 0, 4);
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::ITERATIVE_SCHUR;
  options.minimizer_progress_to_stdout = true;
//  options.logging_type = log_type;
  //options.max_num_iterations = 1000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  MLOG() << summary.BriefReport();
  LOG(ERROR) << summary.FullReport();
  LOG(ERROR) << "result values = " << r;

  cv::Matx33d cam_R_car;
  cv::Rodrigues(r, cam_R_car);
  cv::Matx33d car_R_cam = cam_R_car.t();

  LOG(ERROR) << std::endl << cam_R_car;

  camera_ptr->set_Car_R_Cam(car_R_cam);
  camera_ptr->set_camera_height(4);
  {
    double residuals[8];
    for (int i = 0; i < lines.size(); i++) {
      LOG(ERROR) << "line[" << i << "]:";
      for (int j = 1; j < lines[i].size(); j++) {
        auto p = camera_ptr->GetPoint3d(lines[i][j]);
        cost_functions[i * 79 + j](r.val, zms[i * 80].data(), zms[i * 80 + j].data(), residuals);
        LOG(ERROR) << p << lines[i][j] << "," << residuals[0] << "," << residuals[1]
                   << "," << residuals[2] << "," << residuals[3] << "," << residuals[4]
                   << "," << residuals[5] << "," << residuals[6] << "," << residuals[7] << ","
                   << zms[i * 80][0] << "," << zms[i * 80][1] << "," << zms[i * 80][0] - zms[i * 80 + j][0] << ","
                   << zms[i * 80 + j][0] << "," << zms[i * 80 + j][1] << "," << zms[i * 80][1] - zms[i * 80 + j][1] << std::endl;
      }
    }
  }

}

int main(int argc, char **argv) {
  google::SetVersionString("1.0.0");
  google::SetUsageMessage(std::string(argv[0]) + " [OPTION]");
  google::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  test_b();

  return 0;

}