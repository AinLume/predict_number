#pragma once 

#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>
#include <iostream>

class Perceptron {
private:
    std::vector<std::vector<double>> weights_in;
    std::vector<std::vector<double>> weights_out;
    double learn_rate = 0.01;

    double sigmoid(double x);
    double sigmoidDerivative(double x);
public:
    Perceptron(int input_size, int hidden_size, int output_size);
    void initWeights(std::vector<std::vector<double>>& weights, int rows, int cols);
    void train(std::vector<std::vector<int>> train_input, std::vector<std::vector<int>> train_output, int epochs);
    std::vector<double> predict(std::vector<int> input);
    void saveWeights();
    void loadWeights();
    void printWeights();
};