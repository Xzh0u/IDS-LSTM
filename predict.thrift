namespace py predict

service Predictor {
    void ping(),

    list<double> pong(1:list<double> data),

    list<double> predict(1:list<double> data),
}
