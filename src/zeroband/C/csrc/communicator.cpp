#include <pybind11/pybind11.h>   // For py::bytes and GIL handling
#include "SocketCommunicator.h"

namespace py = pybind11;

PYBIND11_MODULE(communicator, m) {
    py::class_<SocketCommunicator>(m, "SocketCommunicator")
        .def(py::init<const std::string&, unsigned short>(),
             py::arg("listen_address"), py::arg("listen_port"))
        .def("setTarget", &SocketCommunicator::setTarget,
             py::arg("target_address"), py::arg("target_port"))
        .def("sendData", &SocketCommunicator::sendData,
             py::arg("data"))
        .def("setDataCallback", [](SocketCommunicator& self, py::function func) {
            // Keep a reference to the Python function
            auto func_ref = std::make_shared<py::function>(func);
            self.setDataCallback([func_ref](const std::string& data) {
                // Acquire GIL before calling the Python callback
                py::gil_scoped_acquire acquire;
                try {
                    (*func_ref)(py::bytes(data));
                } catch (py::error_already_set& e) {
                    // Print the exception and continue
                    PyErr_Print();
                }
            });
        }, py::arg("callback"));
}
