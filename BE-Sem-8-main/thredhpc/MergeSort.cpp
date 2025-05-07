#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

void print_array(const vector<int>& arr) {
    for (int val : arr)
        cout << val << " ";
    cout << endl;
}

void merge(vector<int>& arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];
}

void merge_sort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        merge_sort(arr, l, m);
        merge_sort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}
void parallel_merge_sort_util(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        #pragma omp task shared(arr)
        parallel_merge_sort_util(arr, l, m);

        #pragma omp task shared(arr)
        parallel_merge_sort_util(arr, m + 1, r);

        #pragma omp taskwait
        merge(arr, l, m, r);
    }
}


void parallel_merge_sort(vector<int>& arr) {
    #pragma omp parallel
    {
        #pragma omp single
        parallel_merge_sort_util(arr, 0, arr.size() - 1);
    }
}

int main() {
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    srand(time(0));
    vector<int> original(n);
    for (int i = 0; i < n; ++i)
        original[i] = rand() % 1501;

    cout << "\nGenerated array: ";
    print_array(original);

    vector<int> arr1 = original;
    vector<int> arr2 = original;

    double start, end;

    // Sequential merge sort
    start = omp_get_wtime();
    merge_sort(arr1, 0, arr1.size() - 1);
    end = omp_get_wtime();
    cout << "\nSequential Merge Sort Output: ";
    print_array(arr1);
    cout << "Time: " << end - start << " seconds\n";

    // Parallel merge sort
    start = omp_get_wtime();
    parallel_merge_sort(arr2);
    end = omp_get_wtime();
    cout << "\nParallel Merge Sort Output: ";
    print_array(arr2);
    cout << "Time: " << end - start << " seconds\n";

    return 0;
}

