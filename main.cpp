#include"base.h"

void formatPoints(std::vector<std::pair<float, float>> points, std::vector<std::pair<float, float>>& format_points)
{
    std::sort(points.begin(), points.end(),
        [](std::pair<float, float>& a, const std::pair<float, float>& b) { return a.first < b.first; });

    if (points[0].second <= points[1].second)
    {
        format_points[0] = points[0];
        format_points[3] = points[1];
    }
    else
    {
        format_points[0] = points[1];
        format_points[3] = points[0];
    }

    if (points[2].second <= points[3].second)
    {
        format_points[1] = points[2];
        format_points[2] = points[3];
    }
    else
    {
        format_points[1] = points[3];
        format_points[2] = points[2];
    }

    return;
}

bool isNearOpenCvResults(std::vector<float> opencvRes, std::vector<float> cvcudaRes)
{
    std::vector<std::pair<float, float>> goldVec{
        std::make_pair(opencvRes[0], opencvRes[1]), std::make_pair(opencvRes[2], opencvRes[3]),
        std::make_pair(opencvRes[4], opencvRes[5]), std::make_pair(opencvRes[6], opencvRes[7]) };
    std::vector<std::pair<float, float>> predVec{
        std::pair<float, float>{cvcudaRes[0], cvcudaRes[1]},
         std::pair<float, float>{cvcudaRes[2], cvcudaRes[3]},
        std::pair<float, float>{cvcudaRes[4], cvcudaRes[5]},
         std::pair<float, float>{cvcudaRes[6], cvcudaRes[7]}
    };
    std::vector<std::pair<float, float>> goldVec_format(4, std::pair<float, float>{0, 0});
    std::vector<std::pair<float, float>> predVec_format(4, std::pair<float, float>{0, 0});
    formatPoints(goldVec, goldVec_format);
    formatPoints(predVec, predVec_format);
    
    for (size_t i = 0; i < predVec_format.size(); i++)
    {
        if (std::abs(goldVec_format[i].first - predVec_format[i].first) > 0.0
            || std::abs(goldVec_format[i].second - predVec_format[i].second) > 0.0)
        {
            return false;
        }
    }
    return true;
}
//------------------------------------------------
void SeparatCoordinate(std::vector<std::vector<int>> param_input_coord, std::vector<std::vector<cv::Point>>& param_out_point) {

    for (int i = 0; i < param_input_coord.size(); i++) {
        std::vector<cv::Point> temp_point_vector;
        for (int j = 0; j < param_input_coord[i].size() - 1; j += 2) {

            cv::Point temp_point = cv::Point(param_input_coord[i][j], param_input_coord[i][j + 1]);
            temp_point_vector.push_back(temp_point);
        }
        param_out_point.push_back(temp_point_vector);
    }
}

void MinRectPoint(std::vector<std::vector<cv::Point>> param_in_vec_point, std::vector<std::vector<cv::Point2f>>& param_out_point) {

    for (int i = 0; i < param_in_vec_point.size(); i++) {

        cv::RotatedRect min_rect = cv::minAreaRect(param_in_vec_point[i]);
        cv::Point2f vertices[4];
        min_rect.points(vertices);

        std::vector<cv::Point2f> vec_conner_point;
        for (int j = 0; j < 4; j++) {
            vec_conner_point.push_back(vertices[j]);
        }
        param_out_point.push_back(vec_conner_point);
    }
}

void PrintMinRectConnerCoord(std::vector<std::vector<cv::Point2f>> param_in_vec_point, std::vector<std::vector<float>>& pram_vec_outer_point) {

    for (int i = 0; i < param_in_vec_point.size(); i++) {

        std::vector<float> temp_coord_point_val;
        for (int j = 0; j < param_in_vec_point[i].size(); j++) {

            temp_coord_point_val.push_back(param_in_vec_point[i][j].x);
            temp_coord_point_val.push_back(param_in_vec_point[i][j].y);
           // std::cout << param_in_vec_point[i][j].x << "  " << param_in_vec_point[i][j].y;
            printf("%.2f  %.2f", param_in_vec_point[i][j].x, param_in_vec_point[i][j].y);
            if (j != param_in_vec_point[i].size() - 1)
                std::cout << "  ";
        }

        pram_vec_outer_point.push_back(temp_coord_point_val);
        std::cout << std::endl;
    }
}

int main()
{

    int batchsize = 3;
    std::vector<std::vector<int>> contourPointsData;
   /* contourPointsData.push_back(
        { 845, 600, 845, 601, 847, 603, 859, 603, 860, 604, 865, 604, 866, 603, 867, 603, 868, 602, 868, 601, 867, 600 });*/
    contourPointsData.push_back(
        { 100,200,100,400,300,200,300,400 });
    contourPointsData.push_back({ 965,  489, 964,  490, 963,  490, 962,  491, 962,  494, 963,  495,
                                 963,  499, 964,  500, 964,  501, 966,  503, 1011, 503, 1012, 504,
                                 1013, 503, 1027, 503, 1027, 502, 1028, 501, 1028, 490, 1027, 489 });
    contourPointsData.push_back({ 1050, 198, 1049, 199, 1040, 199, 1040, 210, 1041, 211, 1040, 212, 1040, 214, 1045, 214,
                                 1046, 213, 1049, 213, 1050, 212, 1051, 212, 1052, 211, 1053, 211, 1054, 210, 1055, 210,
                                 1056, 209, 1058, 209, 1059, 208, 1059, 200, 1058, 200, 1057, 199, 1051, 199 });
                                 
    std::vector<std::vector<float>> CUDA_minAreaRect_results(batchsize, std::vector<float>(8, 0));
    std::vector<std::vector<cv::Point>> contourPointsDataH;

    for (int ci = 0; ci < batchsize; ci++)
    {
        auto contour = contourPointsData[ci];
        std::vector<cv::Point>points;
        for (int i = 0; i < contour.size(); i += 2)
        {
            points.push_back(cv::Point(contour[i], contour[i + 1]));
        }
        contourPointsDataH.push_back(points);
    }
    std::cout << "CUDA_minAreaRect_results" << std::endl;
    MinAreaRectF(contourPointsDataH, CUDA_minAreaRect_results);//得到最小截矩形的四个点
   

    //--------------以下为新加代码---------------------
    std::cout <<std::endl<< "openCV_minAreaRect_results" << std::endl;
    std::vector<std::vector<cv::Point>> seprate_coord;
    SeparatCoordinate(contourPointsData, seprate_coord);//把int型转化为point型

    std::vector<std::vector<cv::Point2f>> outer_coord;
    MinRectPoint(seprate_coord, outer_coord);

    std::vector<std::vector<float>> openCV_minAreaRect_results;
    PrintMinRectConnerCoord(outer_coord, openCV_minAreaRect_results);
    for (int ci = 0; ci < batchsize; ci++)
    {
        auto passOrNo = isNearOpenCvResults(openCV_minAreaRect_results[ci], CUDA_minAreaRect_results[ci]);
        printf("%s\n", passOrNo == true ? "pass" : "no");
    }
}                               
//cv2.minAreaRect函数的返回值是旋转矩形框中心点的坐标和框的宽高，想要在图像画出旋转矩形框还需要将其转成4个顶点的形式