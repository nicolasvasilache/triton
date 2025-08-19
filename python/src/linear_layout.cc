#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "triton/Tools/LinearLayout.h"

using namespace mlir;
namespace py = pybind11;
namespace tt = triton;

void defineLinearLayout(py::module &m) {
  // Expose LinearLayout class
  py::class_<tt::LinearLayout>(m, "LinearLayout", py::module_local())
      .def("__eq__", [](const tt::LinearLayout &self, const tt::LinearLayout &other) {
          return self.equalIgnoringOutDimSizes(other);
      }, "Check if two layouts are equal")
      .def("__ne__", [](const tt::LinearLayout &self, const tt::LinearLayout &other) {
          return !self.equalIgnoringOutDimSizes(other);
      }, "Check if two layouts are not equal")
      .def("__mul__", [](const tt::LinearLayout& inner, const tt::LinearLayout& outer) {
          return inner * outer;
      }, "Direct sum of two layouts: inner * outer")
      .def("__imul__", [](tt::LinearLayout& self, const tt::LinearLayout& other) -> tt::LinearLayout& {
          return self *= other;
      }, "In-place direct sum: self *= other")
      .def("__str__", &tt::LinearLayout::toString,
           "Get string representation of the layout")
      .def("__repr__", &tt::LinearLayout::toString,
           "Get string representation of the layout")
      .def("to_pretty_binary_string", &tt::LinearLayout::toPrettyBinaryString,
           "Get the binary matrix representation of the layout as a string")
      .def_static("empty", &tt::LinearLayout::empty,
                  "Create a 0-dimensional layout that maps everything to 0")
      .def_static(
          "strided1D",
          [](int size, int stride, const std::string &inDim,
             const std::string &outDim, MLIRContext *ctx) {
            auto inDimAttr = StringAttr::get(ctx, inDim);
            auto outDimAttr = StringAttr::get(ctx, outDim);
            return tt::LinearLayout::strided1D(size, stride, inDimAttr,
                                               outDimAttr);
          },
          "Create a 1D -> 1D layout L(x) = stride * x for x in [0, size)")
      .def_static(
          "identity1D",
          [](int size, const std::string &inDim, const std::string &outDim,
             MLIRContext *ctx) {
            auto inDimAttr = StringAttr::get(ctx, inDim);
            auto outDimAttr = StringAttr::get(ctx, outDim);
            return tt::LinearLayout::identity1D(size, inDimAttr, outDimAttr);
          },
          "Create a 1D -> 1D identity layout L(x) = x for x in [0, size)")
      .def_static(
          "zeros1D",
          [](int size, const std::string &inDim, const std::string &outDim,
             MLIRContext *ctx, int outDimSize = 1) {
            auto inDimAttr = StringAttr::get(ctx, inDim);
            auto outDimAttr = StringAttr::get(ctx, outDim);
            return tt::LinearLayout::zeros1D(size, inDimAttr, outDimAttr,
                                             outDimSize);
          },
          py::arg("size"), py::arg("in_dim"), py::arg("out_dim"),
          py::arg("context"), py::arg("out_dim_size") = 1,
          "Create a 1D -> 1D layout L(x) = 0 for x in [0, size)")
      .def(py::init([](const std::vector<std::pair<
                           std::string, std::vector<std::vector<int>>>> &bases,
                       const std::vector<std::string> &outDimNames,
                       MLIRContext *ctx) {
             // Convert string dimension names to StringAttr
             std::vector<
                 std::pair<StringAttr, std::vector<std::vector<int32_t>>>>
                 mlirBases;
             for (const auto &[dimName, dimBases] : bases) {
               auto dimAttr = StringAttr::get(ctx, dimName);
               std::vector<std::vector<int32_t>> convertedBases;
               for (const auto &basis : dimBases) {
                 std::vector<int32_t> convertedBasis(basis.begin(),
                                                     basis.end());
                 convertedBases.push_back(convertedBasis);
               }
               mlirBases.emplace_back(dimAttr, convertedBases);
             }
             std::vector<StringAttr> mlirOutDimNames;
             for (const auto &name : outDimNames) {
               mlirOutDimNames.push_back(StringAttr::get(ctx, name));
             }
             return tt::LinearLayout(mlirBases, mlirOutDimNames);
           }),
           "Create LinearLayout from explicit bases")
      .def(py::init([](const std::vector<std::pair<
                           std::string, std::vector<std::vector<int>>>> &bases,
                       const std::vector<std::pair<std::string, int>> &outDims,
                       bool requireSurjective,
                       MLIRContext *ctx) {
             // Convert string dimension names to StringAttr
             std::vector<
                 std::pair<StringAttr, std::vector<std::vector<int32_t>>>>
                 mlirBases;
             for (const auto &[dimName, dimBases] : bases) {
               auto dimAttr = StringAttr::get(ctx, dimName);
               std::vector<std::vector<int32_t>> convertedBases;
               for (const auto &basis : dimBases) {
                 std::vector<int32_t> convertedBasis(basis.begin(),
                                                     basis.end());
                 convertedBases.push_back(convertedBasis);
               }
               mlirBases.emplace_back(dimAttr, convertedBases);
             }
             std::vector<std::pair<StringAttr, int32_t>> mlirOutDims;
             for (const auto &[name, size] : outDims) {
               mlirOutDims.emplace_back(StringAttr::get(ctx, name), size);
             }
             return tt::LinearLayout(mlirBases, mlirOutDims, requireSurjective);
           }),
           py::arg("bases"), py::arg("out_dims"), py::arg("require_surjective"), py::arg("context"),
           "Create LinearLayout from explicit bases with output dimensions and surjective requirement")
      .def("get_num_in_dims", &tt::LinearLayout::getNumInDims,
           "Get the number of input dimensions")
      .def("get_num_out_dims", &tt::LinearLayout::getNumOutDims,
           "Get the number of output dimensions")
      .def(
          "has_in_dim",
          [](const tt::LinearLayout &self, const std::string &dimName,
             MLIRContext *ctx) {
            auto dimAttr = StringAttr::get(ctx, dimName);
            return self.hasInDim(dimAttr);
          },
          "Check if the layout has the specified input dimension")
      .def(
          "has_out_dim",
          [](const tt::LinearLayout &self, const std::string &dimName,
             MLIRContext *ctx) {
            auto dimAttr = StringAttr::get(ctx, dimName);
            return self.hasOutDim(dimAttr);
          },
          "Check if the layout has the specified output dimension")
      .def(
          "get_in_dim_size",
          [](const tt::LinearLayout &self, const std::string &dimName,
             MLIRContext *ctx) {
            auto dimAttr = StringAttr::get(ctx, dimName);
            return self.getInDimSize(dimAttr);
          },
          "Get the size of the specified input dimension")
      .def(
          "get_out_dim_size",
          [](const tt::LinearLayout &self, const std::string &dimName,
             MLIRContext *ctx) {
            auto dimAttr = StringAttr::get(ctx, dimName);
            return self.getOutDimSize(dimAttr);
          },
          "Get the size of the specified output dimension")
      .def(
          "apply",
          [](const tt::LinearLayout &self,
             const std::vector<std::pair<std::string, int>> &inputs,
             MLIRContext *ctx) {
            // Convert Python inputs to MLIR format
            std::vector<std::pair<StringAttr, int32_t>> mlirInputs;
            for (const auto &[dimName, value] : inputs) {
              auto dimAttr = StringAttr::get(ctx, dimName);
              mlirInputs.emplace_back(dimAttr, value);
            }
            // Apply the layout
            auto result = self.apply(mlirInputs);
            // Convert result back to Python format
            std::vector<std::pair<std::string, int>> pythonResult;
            for (const auto &[dimAttr, value] : result) {
              pythonResult.emplace_back(dimAttr.str(), value);
            }
            return pythonResult;
          },
          "Apply the layout function L(inputs) and return the output values")
             .def("compose", &tt::LinearLayout::compose,
            "Compose this layout with another layout: (outer ∘ this)(x) = "
            "outer(this(x))")
       .def("is_surjective", &tt::LinearLayout::isSurjective,
            "Check if the layout is surjective")
       .def("is_injective", &tt::LinearLayout::isInjective,
            "Check if the layout is injective")
       .def("is_invertible", &tt::LinearLayout::isInvertible,
            "Check if the layout is invertible")
       .def(
           "unsqueeze_in",
           [](const tt::LinearLayout &self, const std::string &dimName,
              MLIRContext *ctx) {
             auto dimAttr = StringAttr::get(ctx, dimName);
             return self.unsqueezeIn(dimAttr);
           },
           "Remove a dimension of size 1 from the input")
       .def(
           "unsqueeze_out",
           [](const tt::LinearLayout &self, const std::string &dimName,
              MLIRContext *ctx) {
             auto dimAttr = StringAttr::get(ctx, dimName);
             return self.unsqueezeOut(dimAttr);
           },
           "Remove a dimension of size 1 from the output")
       .def(
           "get_basis",
           [](const tt::LinearLayout &self, const std::string &inDim,
              int32_t pos, MLIRContext *ctx) {
             auto inDimAttr = StringAttr::get(ctx, inDim);
             auto basis = self.getBasis(inDimAttr, pos);
             std::vector<int32_t> result(basis.begin(), basis.end());
             return result;
           },
           "Get the pos'th basis vector for the inDim -> outDim mapping")
       .def(
           "get_basis_component",
           [](const tt::LinearLayout &self, const std::string &inDim,
              int32_t pos, const std::string &outDim, MLIRContext *ctx) {
             auto inDimAttr = StringAttr::get(ctx, inDim);
             auto outDimAttr = StringAttr::get(ctx, outDim);
             return self.getBasis(inDimAttr, pos, outDimAttr);
           },
           "Get the component of the pos'th basis vector for inDim -> outDim mapping")
       .def(
           "get_out_dims",
           [](const tt::LinearLayout &self) {
             auto outDims = self.getOutDims();
             std::vector<std::pair<std::string, int32_t>> result;
             for (const auto &[dimAttr, size] : outDims) {
               result.emplace_back(dimAttr.str(), size);
             }
             return result;
           },
           "Get all output dimensions and their sizes as a list of (name, size) pairs")
       .def(
           "transpose_ins",
           [](const tt::LinearLayout &self,
              const std::vector<std::string> &newInDimOrder,
              MLIRContext *ctx) {
             std::vector<StringAttr> newInDimAttrs;
             for (const auto &name : newInDimOrder) {
               newInDimAttrs.push_back(StringAttr::get(ctx, name));
             }
             return self.transposeIns(newInDimAttrs);
           },
           "Reorder the input dimensions of the layout")
       .def(
           "transpose_outs",
           [](const tt::LinearLayout &self,
              const std::vector<std::string> &newOutDimOrder,
              MLIRContext *ctx) {
             std::vector<StringAttr> newOutDimAttrs;
             for (const auto &name : newOutDimOrder) {
               newOutDimAttrs.push_back(StringAttr::get(ctx, name));
             }
             return self.transposeOuts(newOutDimAttrs);
           },
           "Reorder the output dimensions of the layout")
       .def(
           "reshape_ins",
           [](const tt::LinearLayout &self,
              const std::vector<std::pair<std::string, int32_t>> &newInDims,
              MLIRContext *ctx) {
             std::vector<std::pair<StringAttr, int32_t>> newInDimAttrs;
             for (const auto &[name, size] : newInDims) {
               newInDimAttrs.emplace_back(StringAttr::get(ctx, name), size);
             }
             return self.reshapeIns(newInDimAttrs);
           },
           "Reshape the input dimensions of the layout")
       .def(
           "reshape_outs",
           [](const tt::LinearLayout &self,
              const std::vector<std::pair<std::string, int32_t>> &newOutDims,
              MLIRContext *ctx) {
             std::vector<std::pair<StringAttr, int32_t>> newOutDimAttrs;
             for (const auto &[name, size] : newOutDims) {
               newOutDimAttrs.emplace_back(StringAttr::get(ctx, name), size);
             }
             return self.reshapeOuts(newOutDimAttrs);
           },
           "Reshape the output dimensions of the layout")
       .def("invert_and_compose", &tt::LinearLayout::invertAndCompose,
            "Compute the inverse of this layout and compose it with another layout")
       .def("invert", &tt::LinearLayout::invert,
            "Get the layout that is the inverse of this layout")
       .def("pseudoinvert", &tt::LinearLayout::pseudoinvert,
            "Compute and return a pseudoinverse of this layout")
      // Omit these for now as they don't give a clear benefit over the more general reshape methods.
      //  .def("flatten_ins", &tt::LinearLayout::flattenIns,
      //       "Reshape to a single input dimension")
      //  .def("flatten_outs", &tt::LinearLayout::flattenOuts,
      //       "Reshape to a single output dimension")
            ;
}
