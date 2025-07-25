# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor
import operator

aten = torch.ops.aten
prims = torch.ops.prims

from torch._inductor.pattern_matcher import (
   Arg,
   CallFunction,
   CallFunctionVarArgs,
   CallMethod,
   CallMethodVarArgs,
   CallModule,
   CallModuleVarArgs,
   ExclusiveKeywordArg,
   Ignored,
   KeywordArg,
   ListOf,
   MultiOutputPattern,
   PatternExpr,
   RepeatedExpr,
   _TargetArgsExpr,
   _TargetExpr,
   _TargetExprVarArgs,
)
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
view_default_3 = CallFunction(aten.view.default, view_default_2, Ignored())
view_default_4 = CallFunction(aten.view.default, view_default_3, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, view_default_4, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_4, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
expand_default_2 = CallFunction(aten.expand.default, div_Tensor, Ignored())
view_default_5 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_6 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, view_default_5, view_default_6)
view_default_7 = CallFunction(aten.view.default, bmm_default_1, Ignored())
neg_default = CallFunction(aten.neg.default, div_Tensor)
view_default_8 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
permute_default_4 = CallFunction(aten.permute.default, view_default_6, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, view_default_8, permute_default_4)
view_default_9 = CallFunction(aten.view.default, bmm_default_2, Ignored())
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_9, div_Tensor, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
view_default_10 = CallFunction(aten.view.default, fma_default, Ignored())
view_default_11 = CallFunction(aten.view.default, view_default_10, Ignored())
view_default_12 = CallFunction(aten.view.default, view_default_11, Ignored(), _users=2)
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, view_default_12, permute_default_5)
view_default_13 = CallFunction(aten.view.default, bmm_default_3, Ignored())
permute_default_6 = CallFunction(aten.permute.default, view_default_13, Ignored())
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_12)
view_default_14 = CallFunction(aten.view.default, bmm_default_4, Ignored())
permute_default_8 = CallFunction(aten.permute.default, view_default_14, Ignored())
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
permute_default_10 = CallFunction(aten.permute.default, view_default_5, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_8)
view_default_15 = CallFunction(aten.view.default, bmm_default_5, Ignored())
permute_default_11 = CallFunction(aten.permute.default, view_default_15, Ignored())
_sfdp_pattern_23_training = MultiOutputPattern([view_default_7,
  permute_default_1,
  permute_default_3,
  permute_default_6,
  permute_default_9,
  permute_default_11
])


permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
view_default_3 = CallFunction(aten.view.default, view_default_2, Ignored())
view_default_4 = CallFunction(aten.view.default, view_default_3, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, view_default_4, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_4, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
expand_default_2 = CallFunction(aten.expand.default, div_Tensor, Ignored())
view_default_5 = CallFunction(aten.view.default, expand_default_2, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_6 = CallFunction(aten.view.default, clone_default_2, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_5, view_default_6)
view_default_7 = CallFunction(aten.view.default, bmm_default_1, Ignored())
_sfdp_pattern_23_inference = MultiOutputPattern([view_default_7,
  permute_default_1,
  permute_default_3
])


permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored())
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, convert_element_type_default, Ignored())
view_default_3 = CallFunction(aten.view.default, convert_element_type_default_1, Ignored())
view_default_4 = CallFunction(aten.view.default, view_default_3, Ignored())
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_4, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default_2, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default_2, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_3, Ignored())
view_default_5 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_6 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, view_default_5, view_default_6)
view_default_7 = CallFunction(aten.view.default, bmm_default_1, Ignored())
neg_default = CallFunction(aten.neg.default, div_Tensor)
view_default_8 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
permute_default_4 = CallFunction(aten.permute.default, view_default_6, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, view_default_8, permute_default_4)
view_default_9 = CallFunction(aten.view.default, bmm_default_2, Ignored())
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, view_default_9, Ignored())
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_4, div_Tensor, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
view_default_10 = CallFunction(aten.view.default, convert_element_type_default_5, Ignored())
view_default_11 = CallFunction(aten.view.default, view_default_10, Ignored())
convert_element_type_default_6 = CallFunction(prims.convert_element_type.default, view_default_11, Ignored())
convert_element_type_default_7 = CallFunction(prims.convert_element_type.default, convert_element_type_default_6, Ignored())
view_default_12 = CallFunction(aten.view.default, convert_element_type_default_7, Ignored(), _users=2)
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, view_default_12, permute_default_5)
view_default_13 = CallFunction(aten.view.default, bmm_default_3, Ignored())
permute_default_6 = CallFunction(aten.permute.default, view_default_13, Ignored())
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_12)
view_default_14 = CallFunction(aten.view.default, bmm_default_4, Ignored())
permute_default_8 = CallFunction(aten.permute.default, view_default_14, Ignored())
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
permute_default_10 = CallFunction(aten.permute.default, view_default_5, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_8)
view_default_15 = CallFunction(aten.view.default, bmm_default_5, Ignored())
permute_default_11 = CallFunction(aten.permute.default, view_default_15, Ignored())
_sfdp_pattern_23_half_training = MultiOutputPattern([view_default_7,
  permute_default_1,
  permute_default_3,
  permute_default_6,
  permute_default_9,
  permute_default_11
])


permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored())
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, convert_element_type_default, Ignored())
view_default_3 = CallFunction(aten.view.default, convert_element_type_default_1, Ignored())
view_default_4 = CallFunction(aten.view.default, view_default_3, Ignored())
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_4, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default_2, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default_2, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_3, Ignored())
view_default_5 = CallFunction(aten.view.default, expand_default_2, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored(), _users=2)
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_6 = CallFunction(aten.view.default, clone_default_2, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_5, view_default_6)
view_default_7 = CallFunction(aten.view.default, bmm_default_1, Ignored())
_sfdp_pattern_23_half_inference = MultiOutputPattern([view_default_7,
  permute_default_1,
  permute_default_3
])
