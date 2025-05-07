#include <iostream>
#include <vector>
#include <omp.h>
#include <climits>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()

using namespace std;

void min_reduction(vector<int>& arr) {
    int min_value = INT_MAX;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(min: min_value)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] < min_value) {
            min_value = arr[i];
        }
    }

    double end = omp_get_wtime();
    cout << "Minimum value: " << min_value << " (Time: " << (end - start) << " seconds)" << endl;
}

void max_reduction(vector<int>& arr) {
    int max_value = INT_MIN;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(max: max_value)
    for (int i = 0; i < arr.size(); i++) {
        if (arr[i] > max_value) {
            max_value = arr[i];
        }
    }

    double end = omp_get_wtime();
    cout << "Maximum value: " << max_value << " (Time: " << (end - start) << " seconds)" << endl;
}

void sum_reduction(vector<int>& arr) {
    int sum = 0;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }

    double end = omp_get_wtime();
    cout << "Sum: " << sum << " (Time: " << (end - start) << " seconds)" << endl;
}

void average_reduction(vector<int>& arr) {
    int sum = 0;
    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+: sum)
    for (int i = 0; i < arr.size(); i++) {
        sum += arr[i];
    }

    double end = omp_get_wtime();
    cout << "Average: " << (double)sum / arr.size() << " (Time: " << (end - start) << " seconds)" << endl;
}

int main() {
    int n;
    cout << "Enter size of the array: ";
    cin >> n;

    srand(time(0)); // Seed for random number generation

    vector<int> arr(n);
    for (int i = 0; i < n; ++i) {
        arr[i] = rand() % 1501;  // Random numbers between 0 and 1500
    }

    cout << "Generated array: ";
    for (int x : arr) cout << x << " ";
    cout << "\n\n";

    cout << "Parallel Reduction Operations with Time Complexity:\n";
    min_reduction(arr);
    max_reduction(arr);
    sum_reduction(arr);
    average_reduction(arr);

    return 0;
}

