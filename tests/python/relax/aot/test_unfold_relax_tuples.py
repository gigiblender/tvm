# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
import tvm.testing
from tvm.script import relax as R


def test_single_func_call_packed():
    @tvm.script.ir_module
    class TestModule:
        @R.function
        def dummy_func(some_var: R.Tuple(R.Tensor((5, 7), "float32"), R.Tensor((3, 5), "float32"))):
            return R.Tuple()

        @R.function
        def main(input: R.Tensor((16, 16), "uint8"), output_1: R.Tensor((5, 7), "float32")):
            tsid_11 = R.builtin.alloc_tensor((1, 1), dtype="int8", runtime_device_index=0)
            # Create a tuple
            tuple_0 = R.Tuple(output_1, input)
            # Create another tuple using tuple_0
            tuple_1 = R.Tuple(R.TupleGetItem(tuple_0, 0), tsid_11)
            # Create an alias tuple
            tuple_2 = tuple_1
            # Use tuple_2 in a dummy call
            _1 = R.call_packed("dummy_func", tuple_2, type_args=())
            # _3 = another_func(R.Tuple(R.TupleGetItem(tuple_0, 1), R.TupleGetItem(tuple_2, 0)), tuple_2)
            #
            # _4 = R.call_packed("another_func", R.Tuple(R.TupleGetItem(tuple_0, 1), R.TupleGetItem(tuple_2, 0)), tuple_2, type_args=())
            return R.Tuple()

    @tvm.script.ir_module
    class ExpectedModule:
        @R.function
        def dummy_func(some_var_0: R.Tensor(None, "float32", ndim = 2), some_var_1: R.Tensor(None, "float32", ndim = 2)):
            return R.Tuple()

        @R.function
        def main(input: R.Tensor((16, 16), "uint8"), output_1: R.Tensor((5, 7), "float32")):
            tsid_11 = R.builtin.alloc_tensor((1, 1), dtype="int8", runtime_device_index=0)
            gv2 = output_1
            _1 = R.call_packed("dummy_func", gv2, tsid_11, type_args=())
            return R.Tuple()

    mod = TestModule
    after_mod = tvm.relax.transform.UnfoldRelaxTuples()(mod)

    expected_mod = ExpectedModule
    tvm.ir.assert_structural_equal(after_mod["main"], expected_mod["main"])
    assert len(after_mod["dummy_func"].params) == 2


def test_multiple_func_calls():
    @tvm.script.ir_module
    class TestModule:
        @R.function
        def another_dummy_func(some_tuple: R.Tuple(R.Tensor((5, 7), "float32"), R.Tensor((3, 5), "float32")),
                               another_tuple: R.Tuple(R.Tensor((5, 7), "float32"), R.Tensor((3, 5), "float32"))):
            return R.Tuple()

        @R.function
        def dummy_func(some_tuple: R.Tuple(R.Tensor((5, 7), "float32"), R.Tensor((3, 5), "float32")), some_var: R.Tensor((5, 7), "float32"),
                       another_tuple: R.Tuple(R.Tensor((5, 7), "float32"), R.Tensor((3, 5), "float32"))):
            alloc_tensor = R.builtin.alloc_tensor((1, 1), dtype="int8", runtime_device_index=0)
            tuple_0 = R.Tuple(some_var, alloc_tensor)
            tuple_1 = R.Tuple(R.TupleGetItem(some_tuple, 0), R.TupleGetItem(another_tuple, 1))
            _1 = R.call_packed("another_dummy_func", tuple_0, tuple_1, type_args=())
            return R.Tuple()

        @R.function
        def main(input: R.Tensor((16, 16), "uint8"), output_1: R.Tensor((5, 7), "float32")):
            tsid_11 = R.builtin.alloc_tensor((1, 1), dtype="int8", runtime_device_index=0)
            # Create a tuple
            tuple_0 = R.Tuple(output_1, input)
            # Create another tuple using tuple_0
            tuple_1 = R.Tuple(R.TupleGetItem(tuple_0, 0), tsid_11)
            # Create an alias tuple
            tuple_2 = tuple_1
            # Call using call packed
            _1 = R.call_packed("dummy_func", tuple_2, input, tuple_1, type_args=())
            # Call using a global symbol
            _3 = R.call_packed("another_dummy_func", R.Tuple(R.TupleGetItem(tuple_0, 1), R.TupleGetItem(tuple_2, 0)), tuple_2, type_args=())
            return R.Tuple()

    @tvm.script.ir_module
    class ExpectedModule:
        @R.function
        def another_dummy_func(some_tuple_0: R.Tensor(None, "float32", ndim = 2), some_tuple_1: R.Tensor(None, "float32", ndim = 2), another_tuple_0: R.Tensor(None, "float32", ndim = 2), another_tuple_1: R.Tensor(None, "float32", ndim = 2)):
            return R.Tuple()

        @R.function
        def dummy_func(some_tuple_01: R.Tensor(None, "float32", ndim = 2), some_tuple_11: R.Tensor(None, "float32", ndim = 2), some_var: R.Tensor((5, 7), "float32"), another_tuple_01: R.Tensor(None, "float32", ndim = 2), another_tuple_11: R.Tensor(None, "float32", ndim = 2)):
            # block 0
            alloc_tensor = R.builtin.alloc_tensor((1, 1), dtype="int8", runtime_device_index=0)
            gv2 = some_tuple_01
            gv3 = another_tuple_11
            _1 = R.call_packed("another_dummy_func", some_var, alloc_tensor, gv2, gv3, type_args=())
            return R.Tuple()
                
        @R.function
        def main(input: R.Tensor((16, 16), "uint8"), output_1: R.Tensor((5, 7), "float32")):
            # block 0
            tsid_11 = R.builtin.alloc_tensor((1, 1), dtype="int8", runtime_device_index=0)
            gv2 = output_1
            _1 = R.call_packed("dummy_func", gv2, tsid_11, input, gv2, tsid_11, type_args=())
            gv6 = input
            gv7 = gv2
            _3 = R.call_packed("another_dummy_func", gv6, gv7, gv2, tsid_11, type_args=())
            return R.Tuple()


    mod = TestModule
    after_mod = tvm.relax.transform.UnfoldRelaxTuples()(mod)

    expected_mod = ExpectedModule

    tvm.ir.assert_structural_equal(after_mod["main"], expected_mod["main"])

    assert len(after_mod["dummy_func"].params) == len(expected_mod["dummy_func"].params)
    assert len(after_mod["dummy_func"].body.blocks[0].bindings) == \
           len(expected_mod["dummy_func"].body.blocks[0].bindings)
    assert len(after_mod["another_dummy_func"].params) == len(expected_mod["another_dummy_func"].params)


