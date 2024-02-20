#include"base.h"

#define BLOCK                      32
#define PI                         3.1415926535897932384626433832795
//tl and bt points coords + area + angle
#define _MIN_AREA_EACH_ANGLE_STRID 6
// max rotate degree of contour points
#define _MAX_ROTATE_DEGREES        90

__global__ void calculateRotateCoef(float2* aCoeffs, const int degrees)
{
    int angle = blockIdx.x * blockDim.x + threadIdx.x;
    if (angle < degrees)
    {
        aCoeffs[angle].x = cos((float)angle *PI );
        aCoeffs[angle].y = sin((float)angle *PI );
    }
}

//
__global__ void resetRotatedPointsBuf(int* rotatedPointsTensor, const int numOfDegrees)
{
    // int pointIdx    = blockIdx.x * blockDim.x + threadIdx.x;
    int contourIdx = blockIdx.x;
    int angleIdx = threadIdx.x;
    if (angleIdx < numOfDegrees)
    {
        auto rotatedPointsTensorp = rotatedPointsTensor + contourIdx * _MAX_ROTATE_DEGREES * _MIN_AREA_EACH_ANGLE_STRID + angleIdx * _MIN_AREA_EACH_ANGLE_STRID;

        rotatedPointsTensorp[0] = INT_MAX;
        rotatedPointsTensorp[1] = INT_MAX;
        rotatedPointsTensorp[2] = INT_MIN;
        rotatedPointsTensorp[3] = INT_MIN;
        rotatedPointsTensorp[4] = -1;
        rotatedPointsTensorp[5] = -1;
    }
}

__global__ void calculateRotateArea(int2* inContourPointsData,
    int* rotatedPointsTensor, float2* rotateCoeffs,
    int* numPointsInContourBuf, int maxNumPointsInContour)
{
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int contourIdx = blockIdx.y;

    int angleIdx = blockIdx.z;
    extern __shared__ float rotateCoeffs_sm[];
    rotateCoeffs_sm[2 * angleIdx] = rotateCoeffs[angleIdx].x;
    rotateCoeffs_sm[2 * angleIdx + 1] = rotateCoeffs[angleIdx].y;

    __syncthreads();

    if (pointIdx < numPointsInContourBuf[contourIdx])
    {
        auto rotatedPointsTensorp = rotatedPointsTensor + contourIdx * _MAX_ROTATE_DEGREES * _MIN_AREA_EACH_ANGLE_STRID + angleIdx * _MIN_AREA_EACH_ANGLE_STRID;

        auto px = inContourPointsData[contourIdx * maxNumPointsInContour + pointIdx].x;
        auto py = inContourPointsData[contourIdx * maxNumPointsInContour + pointIdx].y;
        float cos_coeff = rotateCoeffs_sm[2 * angleIdx];
        float sin_coeff = rotateCoeffs_sm[2 * angleIdx + 1];
        int   px_rot = (px * cos_coeff) - (py * sin_coeff);
        int   py_rot = (px * sin_coeff) + (py * cos_coeff);
        //xmin
        atomicMin(&rotatedPointsTensorp[0], px_rot);
        //ymin
        atomicMin(&rotatedPointsTensorp[1], py_rot);
        //xmax
        atomicMax(&rotatedPointsTensorp[2], px_rot);
        //ymax
        atomicMax(&rotatedPointsTensorp[3], py_rot);

        __threadfence();
        int rectWidth
            = rotatedPointsTensorp[2] - rotatedPointsTensorp[0];
        int rectHeight
            = rotatedPointsTensorp[3] - rotatedPointsTensorp[1];
        rotatedPointsTensorp[4] = rectWidth * rectHeight;
        rotatedPointsTensorp[5] = angleIdx;
    }
}


__global__ void findMinAreaAndAngle(int* rotatedPointsTensor, float* outMinAreaRectBox,
    const int numOfDegrees)
{
    int angleIdx = threadIdx.x;
    if (angleIdx > numOfDegrees)
    {
        return;
    }

    int rectIdx = blockIdx.x;
    extern __shared__ int areaAngleBuf_sm[];
    auto rotatedPointsTensorp = rotatedPointsTensor + rectIdx * _MAX_ROTATE_DEGREES * _MIN_AREA_EACH_ANGLE_STRID + angleIdx * _MIN_AREA_EACH_ANGLE_STRID;
    areaAngleBuf_sm[2 * angleIdx] = rotatedPointsTensorp[4];
    areaAngleBuf_sm[2 * angleIdx + 1] = rotatedPointsTensorp[5];
    __syncthreads();

    for (int stride = numOfDegrees / 2; stride > 0; stride >>= 1)
    {
        if (angleIdx < stride)
        {
            int* curAreaIdx = &areaAngleBuf_sm[2 * angleIdx];
            int* nextAreaIdx = &areaAngleBuf_sm[2 * (angleIdx + stride)];
            int* curAngleIdx = &areaAngleBuf_sm[2 * angleIdx + 1];
            int* nextAngleIdx = &areaAngleBuf_sm[2 * (angleIdx + stride) + 1];
            if (*curAreaIdx > *nextAreaIdx)
            {
                *curAreaIdx = *nextAreaIdx;
                *curAngleIdx = *nextAngleIdx;
            }
        }
        __syncthreads();

        if (stride % 2 == 1 && areaAngleBuf_sm[0] > areaAngleBuf_sm[2 * (stride - 1)])
        {
            areaAngleBuf_sm[0] = areaAngleBuf_sm[2 * (stride - 1)];
            areaAngleBuf_sm[1] = areaAngleBuf_sm[2 * (stride - 1) + 1];
        }
        __syncthreads();
    }
    if (numOfDegrees % 2 == 1 && areaAngleBuf_sm[0] > areaAngleBuf_sm[2 * (numOfDegrees - 1)])
    {
        areaAngleBuf_sm[0] = areaAngleBuf_sm[2 * (numOfDegrees - 1)];
        areaAngleBuf_sm[1] = areaAngleBuf_sm[2 * (numOfDegrees - 1) + 1];
    }
    if (threadIdx.x == 0)
    {
        int   minRotateAngle = areaAngleBuf_sm[1];
        float cos_coeff = cos(-minRotateAngle * PI / 180.f);
        float sin_coeff = sin(-minRotateAngle * PI / 180.f);
        
        auto rotatedPointsTensorT= rotatedPointsTensor + rectIdx * _MAX_ROTATE_DEGREES * _MIN_AREA_EACH_ANGLE_STRID + areaAngleBuf_sm[1] * _MIN_AREA_EACH_ANGLE_STRID;
        
        float xmin = rotatedPointsTensorT[0];
        float ymin = rotatedPointsTensorT[1];
        float xmax = rotatedPointsTensorT[2];
        float ymax = rotatedPointsTensorT[3];

        float tl_x = (xmin * cos_coeff) - (ymin * sin_coeff);
        float tl_y = (xmin * sin_coeff) + (ymin * cos_coeff);
        float br_x = (xmax * cos_coeff) - (ymax * sin_coeff);
        float br_y = (xmax * sin_coeff) + (ymax * cos_coeff);
        float tr_x = (xmax * cos_coeff) - (ymin * sin_coeff);
        float tr_y = (xmax * sin_coeff) + (ymin * cos_coeff);
        float bl_x = (xmin * cos_coeff) - (ymax * sin_coeff);
        float bl_y = (xmin * sin_coeff) + (ymax * cos_coeff);

        auto outMinAreaRectBoxP = outMinAreaRectBox + rectIdx * 8;

        outMinAreaRectBoxP[0] = bl_x;
        outMinAreaRectBoxP[1] = bl_y;
        outMinAreaRectBoxP[2] = tl_x;
        outMinAreaRectBoxP[3] = tl_y;
        outMinAreaRectBoxP[4] = tr_x;
        outMinAreaRectBoxP[5] = tr_y;
        outMinAreaRectBoxP[6] = br_x;
        outMinAreaRectBoxP[7] = br_y;
    }
}


void calculateRotateCoefCUDA(float2* rotateCoefBuf, const int degrees, const cudaStream_t& stream)
{
    dim3 block(BLOCK * 8);
    dim3 grid(cv::divUp(degrees, block.x));
    calculateRotateCoef << <grid, block, 0, stream >> > (rotateCoefBuf, degrees);
}


void MinAreaRectF(std::vector<std::vector<cv::Point>>contours, std::vector<std::vector<float>>& result, cudaStream_t stream)
{

    // rotateCoeffsData 每个angle 的 cos 值和 sin 值
    float2* rotateCoeffsData = 0;
    cudaMalloc((void**)&rotateCoeffsData, sizeof(float) * _MAX_ROTATE_DEGREES * 2);
    calculateRotateCoefCUDA(rotateCoeffsData, _MAX_ROTATE_DEGREES, stream);

    // contours内最大点集数
    int maxNumPointsInContour = 0;

    for (int id = 0; id < contours.size(); id++)
    {
        maxNumPointsInContour = std::max(maxNumPointsInContour, (int)contours[id].size());
    }

    // contour个数
    int contourBatch = contours.size();

    // pointsInContourData 保存contour内每层的点集数
    int* numPointsInContourH = (int*)malloc(sizeof(int) * contourBatch);
    // contour内的点集
    int2* inContourPointsDataH = (int2*)malloc(sizeof(int) * maxNumPointsInContour * contourBatch * 2);
    memset(inContourPointsDataH,0,sizeof(int) * maxNumPointsInContour * contourBatch * 2);
    for (int id = 0; id < contours.size(); id++)
    {
        auto contour = contours[id];
        auto inContourPointsDataHP = inContourPointsDataH + id * maxNumPointsInContour;
        for (int pid = 0; pid < contour.size(); pid++)
        {

            inContourPointsDataHP[pid].x = contour[pid].x;
            inContourPointsDataHP[pid].y = contour[pid].y;
        }
        numPointsInContourH[id] = (int)contour.size();
    }
    // pointsInContourData 保存contour内每层的点集数
    int* pointsInContourData = 0;
    cudaMalloc((void**)&pointsInContourData, sizeof(int) * contourBatch);
    cudaMemcpy(pointsInContourData, numPointsInContourH, sizeof(int) * contourBatch, cudaMemcpyHostToDevice);
    // contour内的点集
    int2* inContourPointsData = 0;
    cudaMalloc((void**)&inContourPointsData, sizeof(int) * maxNumPointsInContour * contourBatch * 2);
    cudaMemcpy(inContourPointsData, inContourPointsDataH, sizeof(int) * maxNumPointsInContour * contourBatch * 2, cudaMemcpyHostToDevice);

    // 输出结果
    float* outMinAreaRectData = 0;
    cudaMalloc((void**)&outMinAreaRectData, sizeof(float) * contourBatch * 8);

    int* rotatedPointsTensor = 0;
    cudaMalloc((void**)&rotatedPointsTensor, sizeof(int) * contourBatch * _MAX_ROTATE_DEGREES * _MIN_AREA_EACH_ANGLE_STRID);

    dim3 block1(128);
    dim3 grid1(contourBatch);
    resetRotatedPointsBuf << <grid1, block1, 0, stream >> > (rotatedPointsTensor, _MAX_ROTATE_DEGREES);

    dim3   block2(256);
    dim3   grid2(cv::divUp(maxNumPointsInContour, block2.x), contourBatch, _MAX_ROTATE_DEGREES);
    size_t smem_size = 2 * _MAX_ROTATE_DEGREES * sizeof(float);
    calculateRotateArea << <grid2, block2, smem_size, stream >> > (inContourPointsData, rotatedPointsTensor,
        rotateCoeffsData, pointsInContourData, maxNumPointsInContour);

    cudaStreamSynchronize(stream);

    dim3 grid3(contourBatch);
    findMinAreaAndAngle << <grid3, block2, smem_size, stream >> > (rotatedPointsTensor, outMinAreaRectData,
        _MAX_ROTATE_DEGREES);

    cudaStreamSynchronize(stream);

    float* resultList = (float*)malloc(sizeof(float) * 8 * contourBatch);
    cudaMemcpy(resultList, outMinAreaRectData, sizeof(float) * 8 * contourBatch, cudaMemcpyDeviceToHost);

    for (int cid = 0; cid < contourBatch; cid++)
    {
        auto resultListP = resultList + cid * 8;
        for (int i = 0; i < 8; i++)
        {
            result[cid][i] = resultListP[i];
            printf(" %.2f ", resultListP[i]);
        }
        printf("\n");
    }

    free(numPointsInContourH);
    free(inContourPointsDataH);
    cudaFree(inContourPointsData);
    cudaFree(rotatedPointsTensor);
    cudaFree(outMinAreaRectData);
    cudaFree(rotateCoeffsData);

}





