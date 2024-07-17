#include <Perceptron.h>

double Perceptron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double Perceptron::sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

void Perceptron::initWeights(std::vector<std::vector<double>>& weights, int rows, int cols) {
    weights.resize(rows, std::vector<double>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
}

Perceptron::Perceptron(int input_size, int hidden_size, int output_size) {
    srand(time(0));
    initWeights(weights_in, hidden_size, input_size);
    initWeights(weights_out, output_size, hidden_size);
}

std::vector<double> Perceptron::predict(std::vector<int> input) {
    std::vector<double> hidden_layer_output(weights_in.size());

    for (int i = 0; i < weights_in.size(); ++i) {
        double sum = 0.0;
        for (int j = 0; j < input.size(); ++j) {
            sum += weights_in[i][j] * input[j];
        }
        hidden_layer_output[i] = sigmoid(sum);
    }

    std::vector<double> final_output(weights_out.size());

    for (int i = 0; i < weights_out.size(); ++i) {
        double sum = 0.0;
        for (int j = 0; j < hidden_layer_output.size(); ++j) {
            sum += weights_out[i][j] * hidden_layer_output[j];
        }
        final_output[i] = sigmoid(sum);
    }

    return final_output;
}

void Perceptron::train(std::vector<std::vector<int>> train_input, std::vector<std::vector<int>> train_output, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < train_input.size(); ++i) {
            std::vector<int> input = train_input[i];
            std::vector<double> hidden_layer_output(weights_in.size());
            for (int j = 0; j < weights_in.size(); ++j) {
                double sum = 0.0;
                for (int k = 0; k < input.size(); ++k) {
                    sum += weights_in[j][k] * input[k];
                }
                hidden_layer_output[j] = sigmoid(sum);
            }

            std::vector<double> final_output(weights_out.size());
            for (int j = 0; j < weights_out.size(); ++j) {
                double sum = 0.0;
                for (int k = 0; k < hidden_layer_output.size(); ++k) {
                    sum += weights_out[j][k] * hidden_layer_output[k];
                }
                final_output[j] = sigmoid(sum);
            }

            // Рассчет ошибки на выходе
            std::vector<double> output_errors(final_output.size());
            for (int j = 0; j < final_output.size(); ++j) {
                output_errors[j] = train_output[i][j] - final_output[j];
            }

            // Расчет ошибки на входе
            std::vector<double> hidden_errors(hidden_layer_output.size());
            for (int j = 0; j < hidden_layer_output.size(); ++j) {
                double error = 0.0;
                for (int k = 0; k < output_errors.size(); ++k) {
                    error += output_errors[k] * weights_out[k][j];
                }
                hidden_errors[j] = error * sigmoidDerivative(hidden_layer_output[j]);
            }

            // Поправка на выходном слое
            for (int j = 0; j < weights_out.size(); ++j) {
                for (int k = 0; k < hidden_layer_output.size(); ++k) {
                    weights_out[j][k] += learn_rate * output_errors[j] * hidden_layer_output[k];
                }
            }

            // Поправка весов на входном слое
            for (int j = 0; j < weights_in.size(); ++j) {
                for (int k = 0; k < input.size(); ++k) {
                    weights_in[j][k] += learn_rate * hidden_errors[j] * input[k];
                }
            }
        }
    }
}

void Perceptron::saveWeights() {
    std::ofstream out;
    out.open("weights_in.txt");
    if (out.is_open()) {
        for (int i = 0; i < weights_in.size(); ++i) {
            for (int j = 0; j < weights_in[i].size(); ++j) {
                out << weights_in[i][j] << ';';
            }
            out << '\n';
        }
    }
    out.close();
    
    std::ofstream outo;
    outo.open("weights_out.txt");
    if (outo.is_open()) {
        for (int i = 0; i < weights_out.size(); ++i) {
            for (int j = 0; j < weights_out[i].size(); ++j) {
                outo << weights_out[i][j] << ';';
            }
            outo << '\n';
        }
    }
    outo.close();
}

void Perceptron::loadWeights() {

    std::string ws;
    std::ifstream in("weights_in.txt");

    if (in.is_open()) {
        while (std::getline(in, ws)) {
            std::string buffer = "";
            std::vector<double> temp;
            for (int i = 0; i < ws.length(); ++i) {
                if (ws[i] != ';') {
                    buffer += ws[i];
                }
                else {
                    temp.push_back(std::stod(buffer));
                    buffer = "";
                }
            }
            weights_in.push_back(temp);
        }
    }
    in.close();

    std::ifstream ino("weights_out.txt");

    if (ino.is_open()) {
        while (std::getline(ino, ws)) {
            std::string buffer = "";
            std::vector<double> temp;
            for (int i = 0; i < ws.length(); ++i) {
                if (ws[i] != ';') {
                    buffer += ws[i];
                }
                else {
                    temp.push_back(std::stod(buffer));
                    buffer = "";
                }
            }
            weights_out.push_back(temp);
        }
    }
    ino.close();
}

void Perceptron::printWeights() {
    std::cout << "Weights input:\n";
    for (int i = 0; i < weights_in.size(); ++i) {
        for (int j = 0; j < weights_in[i].size(); ++j) {
            std::cout << weights_in[i][j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "Weights output:\n";
    for (int i = 0; i < weights_out.size(); ++i) {
        for (int j = 0; j < weights_out[i].size(); ++j) {
            std::cout << weights_out[i][j] << ' ';
        }
        std::cout << '\n';
    }
}