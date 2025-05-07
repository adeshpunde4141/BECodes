#include <iostream>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <ctime> 
using namespace std;

// Sequential Bubble Sort
void sequential_bubble_sort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort using Odd-Even Transposition
void bubble_sort_odd_even(vector<int>& arr) {
    bool isSorted = false;
    int n = arr.size();
    while (!isSorted) {
        isSorted = true;

        // Even phase
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }

        // Odd phase
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i += 2) {
            if (arr[i] > arr[i + 1]) {
                swap(arr[i], arr[i + 1]);
                isSorted = false;
            }
        }
    }
}
// Helper function to print array
void print_array(const vector<int>& arr) {
    for (int val : arr)
        cout << val << " ";
    cout << endl;
}

int main() {
   int size;
    cout << "Enter the size of the array: ";
    cin >> size;

    // Seed the random number generator
    srand(time(0));

    // Generate random numbers between 0 and 1500
    vector<int> original;
    for (int i = 0; i < size; ++i) {
        original.push_back(rand() % 1501); // 0 to 1500
    }
    
    cout <<"\nGenerated array: ";
   	print_array(original);
   	
    double start, end;

    // Sequential Bubble Sort
    start = omp_get_wtime();
    sequential_bubble_sort(original);
    end = omp_get_wtime();
    cout << "\n\nSequential Bubble Sort Output: ";
    print_array(original);
    cout << "\nTime: " << end - start << " seconds" << endl;

    // Parallel Bubble Sort (Odd-Even)
    start = omp_get_wtime();
    bubble_sort_odd_even(original);
    end = omp_get_wtime();
    cout << "\n\n\nParallel Bubble Sort (Odd-Even) Output: ";
    print_array(original);
    cout << "\nTime: " << end - start << " seconds" << endl;

    return 0;
}

