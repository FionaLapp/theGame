??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.02unknown8??
z
dense_88/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:F * 
shared_namedense_88/kernel
s
#dense_88/kernel/Read/ReadVariableOpReadVariableOpdense_88/kernel*
_output_shapes

:F *
dtype0
r
dense_88/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_88/bias
k
!dense_88/bias/Read/ReadVariableOpReadVariableOpdense_88/bias*
_output_shapes
: *
dtype0
z
dense_89/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_89/kernel
s
#dense_89/kernel/Read/ReadVariableOpReadVariableOpdense_89/kernel*
_output_shapes

: *
dtype0
r
dense_89/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_89/bias
k
!dense_89/bias/Read/ReadVariableOpReadVariableOpdense_89/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
 

	0

1
2
3
 

	0

1
2
3
?
metrics
	variables
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables

layers
layer_regularization_losses
 
[Y
VARIABLE_VALUEdense_88/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_88/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
?

layers
	variables
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
metrics
layer_regularization_losses
[Y
VARIABLE_VALUEdense_89/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_89/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

layers
	variables
regularization_losses
trainable_variables
 layer_metrics
!non_trainable_variables
"metrics
#layer_regularization_losses
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
{
serving_default_input_45Placeholder*'
_output_shapes
:?????????F*
dtype0*
shape:?????????F
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_45dense_88/kerneldense_88/biasdense_89/kerneldense_89/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_4412739
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_88/kernel/Read/ReadVariableOp!dense_88/bias/Read/ReadVariableOp#dense_89/kernel/Read/ReadVariableOp!dense_89/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_4412873
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_88/kerneldense_88/biasdense_89/kerneldense_89/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_4412895??
?

*__inference_dense_88_layer_call_fn_4412819

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_44126122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
 __inference__traced_save_4412873
file_prefix.
*savev2_dense_88_kernel_read_readvariableop,
(savev2_dense_88_bias_read_readvariableop.
*savev2_dense_89_kernel_read_readvariableop,
(savev2_dense_89_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b624e33f16ff49b5bdcedbb98dea3a58/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_88_kernel_read_readvariableop(savev2_dense_88_bias_read_readvariableop*savev2_dense_89_kernel_read_readvariableop(savev2_dense_89_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: :F : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:F : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?
?
%__inference_signature_wrapper_4412739
input_45
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_45unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_44125972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????F
"
_user_specified_name
input_45
?
?
E__inference_dense_88_layer_call_and_return_conditional_losses_4412810

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:::O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
#__inference__traced_restore_4412895
file_prefix$
 assignvariableop_dense_88_kernel$
 assignvariableop_1_dense_88_bias&
"assignvariableop_2_dense_89_kernel$
 assignvariableop_3_dense_89_bias

identity_5??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_88_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_88_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_89_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_89_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
E__inference_dense_89_layer_call_and_return_conditional_losses_4412829

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412773

inputs+
'dense_88_matmul_readvariableop_resource,
(dense_88_biasadd_readvariableop_resource+
'dense_89_matmul_readvariableop_resource,
(dense_89_biasadd_readvariableop_resource
identity??
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes

:F *
dtype02 
dense_88/MatMul/ReadVariableOp?
dense_88/MatMulMatMulinputs&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_88/MatMul?
dense_88/BiasAdd/ReadVariableOpReadVariableOp(dense_88_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_88/BiasAdd/ReadVariableOp?
dense_88/BiasAddBiasAdddense_88/MatMul:product:0'dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_88/BiasAdds
dense_88/ReluReludense_88/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_88/Relu?
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_89/MatMul/ReadVariableOp?
dense_89/MatMulMatMuldense_88/Relu:activations:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_89/MatMul?
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_89/BiasAdd/ReadVariableOp?
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_89/BiasAddm
IdentityIdentitydense_89/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::::O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
"__inference__wrapped_model_4412597
input_459
5sequential_44_dense_88_matmul_readvariableop_resource:
6sequential_44_dense_88_biasadd_readvariableop_resource9
5sequential_44_dense_89_matmul_readvariableop_resource:
6sequential_44_dense_89_biasadd_readvariableop_resource
identity??
,sequential_44/dense_88/MatMul/ReadVariableOpReadVariableOp5sequential_44_dense_88_matmul_readvariableop_resource*
_output_shapes

:F *
dtype02.
,sequential_44/dense_88/MatMul/ReadVariableOp?
sequential_44/dense_88/MatMulMatMulinput_454sequential_44/dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_44/dense_88/MatMul?
-sequential_44/dense_88/BiasAdd/ReadVariableOpReadVariableOp6sequential_44_dense_88_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_44/dense_88/BiasAdd/ReadVariableOp?
sequential_44/dense_88/BiasAddBiasAdd'sequential_44/dense_88/MatMul:product:05sequential_44/dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_44/dense_88/BiasAdd?
sequential_44/dense_88/ReluRelu'sequential_44/dense_88/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_44/dense_88/Relu?
,sequential_44/dense_89/MatMul/ReadVariableOpReadVariableOp5sequential_44_dense_89_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_44/dense_89/MatMul/ReadVariableOp?
sequential_44/dense_89/MatMulMatMul)sequential_44/dense_88/Relu:activations:04sequential_44/dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_44/dense_89/MatMul?
-sequential_44/dense_89/BiasAdd/ReadVariableOpReadVariableOp6sequential_44_dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_44/dense_89/BiasAdd/ReadVariableOp?
sequential_44/dense_89/BiasAddBiasAdd'sequential_44/dense_89/MatMul:product:05sequential_44/dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_44/dense_89/BiasAdd{
IdentityIdentity'sequential_44/dense_89/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::::Q M
'
_output_shapes
:?????????F
"
_user_specified_name
input_45
?
?
/__inference_sequential_44_layer_call_fn_4412724
input_45
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_45unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_44_layer_call_and_return_conditional_losses_44127132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????F
"
_user_specified_name
input_45
?
?
/__inference_sequential_44_layer_call_fn_4412799

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_44_layer_call_and_return_conditional_losses_44127132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
/__inference_sequential_44_layer_call_fn_4412697
input_45
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_45unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_44_layer_call_and_return_conditional_losses_44126862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????F
"
_user_specified_name
input_45
?
?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412756

inputs+
'dense_88_matmul_readvariableop_resource,
(dense_88_biasadd_readvariableop_resource+
'dense_89_matmul_readvariableop_resource,
(dense_89_biasadd_readvariableop_resource
identity??
dense_88/MatMul/ReadVariableOpReadVariableOp'dense_88_matmul_readvariableop_resource*
_output_shapes

:F *
dtype02 
dense_88/MatMul/ReadVariableOp?
dense_88/MatMulMatMulinputs&dense_88/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_88/MatMul?
dense_88/BiasAdd/ReadVariableOpReadVariableOp(dense_88_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_88/BiasAdd/ReadVariableOp?
dense_88/BiasAddBiasAdddense_88/MatMul:product:0'dense_88/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_88/BiasAdds
dense_88/ReluReludense_88/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_88/Relu?
dense_89/MatMul/ReadVariableOpReadVariableOp'dense_89_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_89/MatMul/ReadVariableOp?
dense_89/MatMulMatMuldense_88/Relu:activations:0&dense_89/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_89/MatMul?
dense_89/BiasAdd/ReadVariableOpReadVariableOp(dense_89_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_89/BiasAdd/ReadVariableOp?
dense_89/BiasAddBiasAdddense_89/MatMul:product:0'dense_89/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_89/BiasAddm
IdentityIdentitydense_89/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F:::::O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?

*__inference_dense_89_layer_call_fn_4412838

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_89_layer_call_and_return_conditional_losses_44126382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
E__inference_dense_88_layer_call_and_return_conditional_losses_4412612

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:F *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????F:::O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
/__inference_sequential_44_layer_call_fn_4412786

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_44_layer_call_and_return_conditional_losses_44126862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412669
input_45
dense_88_4412658
dense_88_4412660
dense_89_4412663
dense_89_4412665
identity?? dense_88/StatefulPartitionedCall? dense_89/StatefulPartitionedCall?
 dense_88/StatefulPartitionedCallStatefulPartitionedCallinput_45dense_88_4412658dense_88_4412660*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_44126122"
 dense_88/StatefulPartitionedCall?
 dense_89/StatefulPartitionedCallStatefulPartitionedCall)dense_88/StatefulPartitionedCall:output:0dense_89_4412663dense_89_4412665*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_89_layer_call_and_return_conditional_losses_44126382"
 dense_89/StatefulPartitionedCall?
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0!^dense_88/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????F
"
_user_specified_name
input_45
?
?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412655
input_45
dense_88_4412623
dense_88_4412625
dense_89_4412649
dense_89_4412651
identity?? dense_88/StatefulPartitionedCall? dense_89/StatefulPartitionedCall?
 dense_88/StatefulPartitionedCallStatefulPartitionedCallinput_45dense_88_4412623dense_88_4412625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_44126122"
 dense_88/StatefulPartitionedCall?
 dense_89/StatefulPartitionedCallStatefulPartitionedCall)dense_88/StatefulPartitionedCall:output:0dense_89_4412649dense_89_4412651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_89_layer_call_and_return_conditional_losses_44126382"
 dense_89/StatefulPartitionedCall?
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0!^dense_88/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????F
"
_user_specified_name
input_45
?
?
E__inference_dense_89_layer_call_and_return_conditional_losses_4412638

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412686

inputs
dense_88_4412675
dense_88_4412677
dense_89_4412680
dense_89_4412682
identity?? dense_88/StatefulPartitionedCall? dense_89/StatefulPartitionedCall?
 dense_88/StatefulPartitionedCallStatefulPartitionedCallinputsdense_88_4412675dense_88_4412677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_44126122"
 dense_88/StatefulPartitionedCall?
 dense_89/StatefulPartitionedCallStatefulPartitionedCall)dense_88/StatefulPartitionedCall:output:0dense_89_4412680dense_89_4412682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_89_layer_call_and_return_conditional_losses_44126382"
 dense_89/StatefulPartitionedCall?
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0!^dense_88/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs
?
?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412713

inputs
dense_88_4412702
dense_88_4412704
dense_89_4412707
dense_89_4412709
identity?? dense_88/StatefulPartitionedCall? dense_89/StatefulPartitionedCall?
 dense_88/StatefulPartitionedCallStatefulPartitionedCallinputsdense_88_4412702dense_88_4412704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_88_layer_call_and_return_conditional_losses_44126122"
 dense_88/StatefulPartitionedCall?
 dense_89/StatefulPartitionedCallStatefulPartitionedCall)dense_88/StatefulPartitionedCall:output:0dense_89_4412707dense_89_4412709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_89_layer_call_and_return_conditional_losses_44126382"
 dense_89/StatefulPartitionedCall?
IdentityIdentity)dense_89/StatefulPartitionedCall:output:0!^dense_88/StatefulPartitionedCall!^dense_89/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????F::::2D
 dense_88/StatefulPartitionedCall dense_88/StatefulPartitionedCall2D
 dense_89/StatefulPartitionedCall dense_89/StatefulPartitionedCall:O K
'
_output_shapes
:?????????F
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
input_451
serving_default_input_45:0?????????F<
dense_890
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?[
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
$__call__
*%&call_and_return_all_conditional_losses
&_default_save_signature"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_44", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_44", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_45"}}, {"class_name": "Dense", "config": {"name": "dense_88", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_89", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 70}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_44", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 70]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_45"}}, {"class_name": "Dense", "config": {"name": "dense_88", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_89", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
'__call__
*(&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_88", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_88", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 70}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 70]}}
?

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
)__call__
**&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_89", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_89", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
"
	optimizer
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
?
metrics
	variables
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables

layers
layer_regularization_losses
$__call__
&_default_save_signature
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
,
+serving_default"
signature_map
!:F 2dense_88/kernel
: 2dense_88/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
?

layers
	variables
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
metrics
layer_regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
!: 2dense_89/kernel
:2dense_89/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

layers
	variables
regularization_losses
trainable_variables
 layer_metrics
!non_trainable_variables
"metrics
#layer_regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
/__inference_sequential_44_layer_call_fn_4412724
/__inference_sequential_44_layer_call_fn_4412786
/__inference_sequential_44_layer_call_fn_4412697
/__inference_sequential_44_layer_call_fn_4412799?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412756
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412773
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412655
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412669?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_4412597?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_45?????????F
?2?
*__inference_dense_88_layer_call_fn_4412819?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_88_layer_call_and_return_conditional_losses_4412810?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_89_layer_call_fn_4412838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_89_layer_call_and_return_conditional_losses_4412829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
5B3
%__inference_signature_wrapper_4412739input_45?
"__inference__wrapped_model_4412597n	
1?.
'?$
"?
input_45?????????F
? "3?0
.
dense_89"?
dense_89??????????
E__inference_dense_88_layer_call_and_return_conditional_losses_4412810\	
/?,
%?"
 ?
inputs?????????F
? "%?"
?
0????????? 
? }
*__inference_dense_88_layer_call_fn_4412819O	
/?,
%?"
 ?
inputs?????????F
? "?????????? ?
E__inference_dense_89_layer_call_and_return_conditional_losses_4412829\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? }
*__inference_dense_89_layer_call_fn_4412838O/?,
%?"
 ?
inputs????????? 
? "???????????
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412655h	
9?6
/?,
"?
input_45?????????F
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412669h	
9?6
/?,
"?
input_45?????????F
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412756f	
7?4
-?*
 ?
inputs?????????F
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_44_layer_call_and_return_conditional_losses_4412773f	
7?4
-?*
 ?
inputs?????????F
p 

 
? "%?"
?
0?????????
? ?
/__inference_sequential_44_layer_call_fn_4412697[	
9?6
/?,
"?
input_45?????????F
p

 
? "???????????
/__inference_sequential_44_layer_call_fn_4412724[	
9?6
/?,
"?
input_45?????????F
p 

 
? "???????????
/__inference_sequential_44_layer_call_fn_4412786Y	
7?4
-?*
 ?
inputs?????????F
p

 
? "???????????
/__inference_sequential_44_layer_call_fn_4412799Y	
7?4
-?*
 ?
inputs?????????F
p 

 
? "???????????
%__inference_signature_wrapper_4412739z	
=?:
? 
3?0
.
input_45"?
input_45?????????F"3?0
.
dense_89"?
dense_89?????????