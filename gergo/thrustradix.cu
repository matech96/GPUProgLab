#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/partition.h>
#include <thrust/extrema.h>

struct bit_mask
{
    const int n;
    bit_mask(int _n) : n(_n) {}
    __host__ __device__
        bool operator()(const int& x)
    {
        return !((x >> n) & 1U);
    }
};

int main()
{
    //Comment this and uncomment next commented lines to run on device
    int A[] = { 2, 36, 8, 11, 5, 20, 55, 1 };
    const int N = sizeof(A) / sizeof(int);
    //thrust::device_vector<int> v(8);
    //v[0] = 2; v[1] = 36; v[2] = 8; v[3] = 11; v[4] = 5; v[5] = 20; v[6] = 55; v[7] = 1;
    
    //Comment this and uncomment next commented lines to run on device
    auto max = *thrust::max_element(thrust::host,A,A+N);
    //auto max_ptr = thrust::max_element(thrust::device, v.begin(), v.end());
    //int max = *max_ptr;
    int bitnum = 0;
    while (max > 0) {
        max = max >> 1;
        ++bitnum;
    }
    
    for (int i = 0; i < bitnum; ++i) {
        //Comment this and uncomment next commented lines to run on device
        thrust::stable_partition(thrust::host, A, A + N, bit_mask(i));
        //thrust::stable_partition(thrust::device, v.begin(), v.end(), bit_mask(i));
    }

    std::cout << "{ 2, 36, 8, 11, 5, 20, 55, 1 } radix thrust sorted:" << std::endl;
    //Comment this and uncomment next commented lines to run on device
    for (int i = 0; i < N; ++i) {
        std::cout << A[i] << " ";
    }
    /*for (auto it = v.begin(); it != v.end(); ++it) {
        std::cout << *it << " ";
    }*/

    return 0;
}
