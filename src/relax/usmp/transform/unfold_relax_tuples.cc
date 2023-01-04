/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

#include "tvm/relax/expr_functor.h"
#include "tvm/tir/function.h"

namespace tvm {
namespace relax {

class UnfoldRelaxTuples : public ExprMutator {

 public:
  explicit UnfoldRelaxTuples(IRModule mod) : mod_(mod) {}

  IRModule operator()() {
    GlobalVar gv = mod_->GetGlobalVar("main");
    auto main_func = Downcast<relax::Function>(mod_->Lookup(gv));
    ICHECK(main_func.defined()) << "Main function is not in the module";

    // Do not visit the main function parameters with this pass.
    {
      Expr new_body = this->VisitExpr(main_func->body);
      Function new_func = Function(main_func->params, new_body, main_func->ret_type,
                                   main_func->ret_shape, main_func->attrs);
      mod_->Update(gv, new_func);
    }


    {
      Map<GlobalVar, BaseFunc> new_funcs;
      // Visit all the other Relax functions in the module.
      for (auto pair : mod_->functions) {
        if (!pair.first.same_as(gv) && pair.second->IsInstance<FunctionNode>()) {
          auto func = Downcast<Function>(pair.second);

          auto new_func = this->VisitExpr(func);
          ICHECK(new_func->IsInstance<FunctionNode>()) << "Expected a FunctionNode.";
          new_funcs.Set(pair.first, Downcast<Function>(new_func));
        }
      }
      for (auto pair : new_funcs) {
        mod_->Update(pair.first, pair.second);
      }
    }

    return mod_;
  }

 private:
  Expr VisitExpr_(const FunctionNode* func_node) override {
    current_scope_ = Scope();

    auto func = GetRef<Function>(func_node);

    auto new_params = Array<Var>();
    for (auto param : func->params) {
      if (param->checked_type()->IsInstance<TupleTypeNode>()) {
        current_scope_.tuple_var_alias_map_[param] = param;
        auto tuple_type = Downcast<TupleType>(param->checked_type());
        int index = 0;
        for (auto field : tuple_type->fields) {
          auto new_param = Var(param->name_hint() + "_" + std::to_string(index),
                               ShapeExpr(), field, field->span);
          new_params.push_back(new_param);
          current_scope_.tuple_vars_[param].push_back(new_param);
          index++;
        }
      } else {
        new_params.push_back(param);
      }
    }

    auto new_body = this->VisitExpr(func_node->body);
    return Function(new_params, new_body, func_node->ret_type, func_node->ret_shape,
                    func_node->attrs);
  }

  Expr VisitExpr_(const CallNode* op) override {
    String func_name;
    if (op->op->IsInstance<ExternFuncNode>() || op->op->IsInstance<GlobalVarNode>()) {
      func_name = op->op->IsInstance<ExternFuncNode>()
                      ? runtime::Downcast<ExternFunc>(op->op)->global_symbol
                      : runtime::Downcast<GlobalVar>(op->op)->name_hint;
    }
    if (!func_name.defined()) {
      return ExprMutator::VisitExpr_(op);
    }
    if (!mod_->ContainGlobalVar(func_name)) {
      return ExprMutator::VisitExpr_(op);
    }

    Array<Expr> new_args = Array<Expr>();
    Array<Type> new_type_args = Array<Type>();
    for (auto arg : op->args) {
      if (arg->IsInstance<VarNode>() && current_scope_.tuple_var_alias_map_
          .find(runtime::Downcast<Var>(arg)) != current_scope_.tuple_var_alias_map_.end()) {
        Var tuple = current_scope_.tuple_var_alias_map_[runtime::Downcast<Var>(arg)];
        for (auto field : current_scope_.tuple_vars_[tuple]) {
          new_args.push_back(field);
        }
      } else if (arg->IsInstance<TupleNode>()) {
        auto tuple = runtime::Downcast<Tuple>(arg);
        for (auto field : tuple->fields) {
          new_args.push_back(field);
        }
      } else {
        new_args.push_back(arg);
      }
    }

    ICHECK(!(mod_->Lookup(func_name)->IsInstance<tir::PrimFuncNode>() &&
        new_args.size() > op->args.size())) << "Can not unfold tuples passed to PrimFuncs: " <<
        op->args.size() << " args expand to " << new_args.size() << " args.";

    if (op->op->IsInstance<ExternFuncNode>() || op->op->IsInstance<GlobalVarNode>()) {
      return Call(op->op, new_args, op->attrs, op->type_args, op->span);
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const TupleGetItemNode* op) override {
    auto var_tuple = Downcast<Var>(op->tuple);
    if (current_scope_.tuple_var_alias_map_.find(var_tuple) !=
        current_scope_.tuple_var_alias_map_.end()) {
      auto actual_tuple = current_scope_.tuple_var_alias_map_[var_tuple];
      return current_scope_.tuple_vars_[actual_tuple][op->index];
    }
    return ExprMutator::VisitExpr_(op);
  }

  void VisitBinding_(const VarBindingNode* binding) final {
    // If this is a binding that creates a new tuple, then delete the binding and register the
    // tuple vars.
    if (auto tuple_node = binding->value.as<relay::TupleNode>()) {
      current_scope_.tuple_var_alias_map_[binding->var] = binding->var;
      for (auto field : tuple_node->fields) {
        if (field->IsInstance<relay::ConstantNode>() || field->IsInstance<VarNode>()) {
          current_scope_.tuple_vars_[binding->var].push_back(field);
        } else {
          LOG(FATAL) << "Tuple field is not a constant or a var";
        }
      }
      return;
    }

    // If this is a binding between two tuples, remove it.
    if (auto tuple_var_node = binding->value.as<VarNode>()) {
      auto tuple_var = GetRef<Var>(tuple_var_node);
      if (current_scope_.tuple_var_alias_map_.find(tuple_var) !=
          current_scope_.tuple_var_alias_map_.end()) {
            current_scope_.tuple_var_alias_map_[binding->var] =
                                    current_scope_.tuple_var_alias_map_[tuple_var];
            return;
      }
    }

    ExprMutator::VisitBinding_(binding);
  }

  IRModule mod_;
  struct Scope {
    Scope() {
      tuple_var_alias_map_.clear();
      tuple_vars_.clear();
    }

    // A map from the tuple var to the vars that are inside the tuple.
    std::unordered_map<Var, Array<Expr>, ObjectHash, ObjectEqual> tuple_vars_;
    // An alias map from a tuple var to the aliased tuple var.
    std::unordered_map<Var, Var, ObjectHash, ObjectEqual> tuple_var_alias_map_;
  };
  Scope current_scope_;
};

}  // namespace relax

namespace transform {

tvm::transform::Pass UnfoldRelaxTuples() {
  auto pass_func = [=](IRModule m, tvm::transform::PassContext ctx) {
    return relax::UnfoldRelaxTuples(m)();
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "relax.usmp.UnfoldRelaxTuples", {});
}

TVM_REGISTER_GLOBAL("relax.transform.UnfoldRelaxTuples").set_body_typed(UnfoldRelaxTuples);

}  // namespace transform
}  // namespace tvm