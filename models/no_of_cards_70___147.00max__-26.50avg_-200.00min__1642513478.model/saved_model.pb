??
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
 ?"serve*2.3.02unknown8??
z
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_40/kernel
s
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes

: *
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
_output_shapes
: *
dtype0
z
dense_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_41/kernel
s
#dense_41/kernel/Read/ReadVariableOpReadVariableOpdense_41/kernel*
_output_shapes

: *
dtype0
r
dense_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_41/bias
k
!dense_41/bias/Read/ReadVariableOpReadVariableOpdense_41/bias*
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
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
 

	0

1
2
3

	0

1
2
3
 
?
trainable_variables
non_trainable_variables
	variables

layers
regularization_losses
metrics
layer_metrics
layer_regularization_losses
 
[Y
VARIABLE_VALUEdense_40/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1

	0

1
 
?
trainable_variables
non_trainable_variables
	variables

layers
regularization_losses
metrics
layer_metrics
layer_regularization_losses
[Y
VARIABLE_VALUEdense_41/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_41/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
non_trainable_variables
	variables

 layers
regularization_losses
!metrics
"layer_metrics
#layer_regularization_losses
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
 
 
?
serving_default_input_21Placeholder*+
_output_shapes
:?????????2*
dtype0* 
shape:?????????2
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_21dense_40/kerneldense_40/biasdense_41/kerneldense_41/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3659639
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOp#dense_41/kernel/Read/ReadVariableOp!dense_41/bias/Read/ReadVariableOpConst*
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
 __inference__traced_save_3659893
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_40/kerneldense_40/biasdense_41/kerneldense_41/bias*
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
#__inference__traced_restore_3659915??
?
?
 __inference__traced_save_3659893
file_prefix.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop.
*savev2_dense_41_kernel_read_readvariableop,
(savev2_dense_41_bias_read_readvariableop
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
value3B1 B+_temp_a28e0c1e32054ac080de8e606a914a19/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableop*savev2_dense_41_kernel_read_readvariableop(savev2_dense_41_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
$: : : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 
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

*__inference_dense_40_layer_call_fn_3659819

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
 *+
_output_shapes
:?????????2 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_36594922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????2::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
#__inference__traced_restore_3659915
file_prefix$
 assignvariableop_dense_40_kernel$
 assignvariableop_1_dense_40_bias&
"assignvariableop_2_dense_41_kernel$
 assignvariableop_3_dense_41_bias

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
AssignVariableOpAssignVariableOp assignvariableop_dense_40_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_40_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_41_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_41_biasIdentity_3:output:0"/device:CPU:0*
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
%__inference_signature_wrapper_3659639
input_21
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_36594572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_21
?
?
/__inference_sequential_20_layer_call_fn_3659597
input_21
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_36595862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_21
?
?
/__inference_sequential_20_layer_call_fn_3659779

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
 *+
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_36596132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?E
?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659753

inputs.
*dense_40_tensordot_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource.
*dense_41_tensordot_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identity??
!dense_40/Tensordot/ReadVariableOpReadVariableOp*dense_40_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_40/Tensordot/ReadVariableOp|
dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_40/Tensordot/axes?
dense_40/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_40/Tensordot/freej
dense_40/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_40/Tensordot/Shape?
 dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_40/Tensordot/GatherV2/axis?
dense_40/Tensordot/GatherV2GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/free:output:0)dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_40/Tensordot/GatherV2?
"dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_40/Tensordot/GatherV2_1/axis?
dense_40/Tensordot/GatherV2_1GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/axes:output:0+dense_40/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_40/Tensordot/GatherV2_1~
dense_40/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_40/Tensordot/Const?
dense_40/Tensordot/ProdProd$dense_40/Tensordot/GatherV2:output:0!dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_40/Tensordot/Prod?
dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_40/Tensordot/Const_1?
dense_40/Tensordot/Prod_1Prod&dense_40/Tensordot/GatherV2_1:output:0#dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_40/Tensordot/Prod_1?
dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_40/Tensordot/concat/axis?
dense_40/Tensordot/concatConcatV2 dense_40/Tensordot/free:output:0 dense_40/Tensordot/axes:output:0'dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/concat?
dense_40/Tensordot/stackPack dense_40/Tensordot/Prod:output:0"dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/stack?
dense_40/Tensordot/transpose	Transposeinputs"dense_40/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_40/Tensordot/transpose?
dense_40/Tensordot/ReshapeReshape dense_40/Tensordot/transpose:y:0!dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_40/Tensordot/Reshape?
dense_40/Tensordot/MatMulMatMul#dense_40/Tensordot/Reshape:output:0)dense_40/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_40/Tensordot/MatMul?
dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_40/Tensordot/Const_2?
 dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_40/Tensordot/concat_1/axis?
dense_40/Tensordot/concat_1ConcatV2$dense_40/Tensordot/GatherV2:output:0#dense_40/Tensordot/Const_2:output:0)dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/concat_1?
dense_40/TensordotReshape#dense_40/Tensordot/MatMul:product:0$dense_40/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_40/Tensordot?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/Tensordot:output:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_40/BiasAddw
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_40/Relu?
!dense_41/Tensordot/ReadVariableOpReadVariableOp*dense_41_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_41/Tensordot/ReadVariableOp|
dense_41/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_41/Tensordot/axes?
dense_41/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_41/Tensordot/free
dense_41/Tensordot/ShapeShapedense_40/Relu:activations:0*
T0*
_output_shapes
:2
dense_41/Tensordot/Shape?
 dense_41/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_41/Tensordot/GatherV2/axis?
dense_41/Tensordot/GatherV2GatherV2!dense_41/Tensordot/Shape:output:0 dense_41/Tensordot/free:output:0)dense_41/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_41/Tensordot/GatherV2?
"dense_41/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_41/Tensordot/GatherV2_1/axis?
dense_41/Tensordot/GatherV2_1GatherV2!dense_41/Tensordot/Shape:output:0 dense_41/Tensordot/axes:output:0+dense_41/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_41/Tensordot/GatherV2_1~
dense_41/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_41/Tensordot/Const?
dense_41/Tensordot/ProdProd$dense_41/Tensordot/GatherV2:output:0!dense_41/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_41/Tensordot/Prod?
dense_41/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_41/Tensordot/Const_1?
dense_41/Tensordot/Prod_1Prod&dense_41/Tensordot/GatherV2_1:output:0#dense_41/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_41/Tensordot/Prod_1?
dense_41/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_41/Tensordot/concat/axis?
dense_41/Tensordot/concatConcatV2 dense_41/Tensordot/free:output:0 dense_41/Tensordot/axes:output:0'dense_41/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_41/Tensordot/concat?
dense_41/Tensordot/stackPack dense_41/Tensordot/Prod:output:0"dense_41/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_41/Tensordot/stack?
dense_41/Tensordot/transpose	Transposedense_40/Relu:activations:0"dense_41/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_41/Tensordot/transpose?
dense_41/Tensordot/ReshapeReshape dense_41/Tensordot/transpose:y:0!dense_41/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_41/Tensordot/Reshape?
dense_41/Tensordot/MatMulMatMul#dense_41/Tensordot/Reshape:output:0)dense_41/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_41/Tensordot/MatMul?
dense_41/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_41/Tensordot/Const_2?
 dense_41/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_41/Tensordot/concat_1/axis?
dense_41/Tensordot/concat_1ConcatV2$dense_41/Tensordot/GatherV2:output:0#dense_41/Tensordot/Const_2:output:0)dense_41/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_41/Tensordot/concat_1?
dense_41/TensordotReshape#dense_41/Tensordot/MatMul:product:0$dense_41/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_41/Tensordot?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/Tensordot:output:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_41/BiasAddq
IdentityIdentitydense_41/BiasAdd:output:0*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2:::::S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
/__inference_sequential_20_layer_call_fn_3659624
input_21
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_21unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_36596132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_21
?
?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659586

inputs
dense_40_3659575
dense_40_3659577
dense_41_3659580
dense_41_3659582
identity?? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_3659575dense_40_3659577*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_36594922"
 dense_40/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_3659580dense_41_3659582*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_36595382"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659613

inputs
dense_40_3659602
dense_40_3659604
dense_41_3659607
dense_41_3659609
identity?? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinputsdense_40_3659602dense_40_3659604*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_36594922"
 dense_40/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_3659607dense_41_3659609*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_36595382"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
/__inference_sequential_20_layer_call_fn_3659766

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
 *+
_output_shapes
:?????????2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_sequential_20_layer_call_and_return_conditional_losses_36595862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659555
input_21
dense_40_3659503
dense_40_3659505
dense_41_3659549
dense_41_3659551
identity?? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinput_21dense_40_3659503dense_40_3659505*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_36594922"
 dense_40/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_3659549dense_41_3659551*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_36595382"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_21
?
?
E__inference_dense_41_layer_call_and_return_conditional_losses_3659538

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????2 :::S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?W
?
"__inference__wrapped_model_3659457
input_21<
8sequential_20_dense_40_tensordot_readvariableop_resource:
6sequential_20_dense_40_biasadd_readvariableop_resource<
8sequential_20_dense_41_tensordot_readvariableop_resource:
6sequential_20_dense_41_biasadd_readvariableop_resource
identity??
/sequential_20/dense_40/Tensordot/ReadVariableOpReadVariableOp8sequential_20_dense_40_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_20/dense_40/Tensordot/ReadVariableOp?
%sequential_20/dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_20/dense_40/Tensordot/axes?
%sequential_20/dense_40/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_20/dense_40/Tensordot/free?
&sequential_20/dense_40/Tensordot/ShapeShapeinput_21*
T0*
_output_shapes
:2(
&sequential_20/dense_40/Tensordot/Shape?
.sequential_20/dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_20/dense_40/Tensordot/GatherV2/axis?
)sequential_20/dense_40/Tensordot/GatherV2GatherV2/sequential_20/dense_40/Tensordot/Shape:output:0.sequential_20/dense_40/Tensordot/free:output:07sequential_20/dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_20/dense_40/Tensordot/GatherV2?
0sequential_20/dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_20/dense_40/Tensordot/GatherV2_1/axis?
+sequential_20/dense_40/Tensordot/GatherV2_1GatherV2/sequential_20/dense_40/Tensordot/Shape:output:0.sequential_20/dense_40/Tensordot/axes:output:09sequential_20/dense_40/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_20/dense_40/Tensordot/GatherV2_1?
&sequential_20/dense_40/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_20/dense_40/Tensordot/Const?
%sequential_20/dense_40/Tensordot/ProdProd2sequential_20/dense_40/Tensordot/GatherV2:output:0/sequential_20/dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_20/dense_40/Tensordot/Prod?
(sequential_20/dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_20/dense_40/Tensordot/Const_1?
'sequential_20/dense_40/Tensordot/Prod_1Prod4sequential_20/dense_40/Tensordot/GatherV2_1:output:01sequential_20/dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_20/dense_40/Tensordot/Prod_1?
,sequential_20/dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_20/dense_40/Tensordot/concat/axis?
'sequential_20/dense_40/Tensordot/concatConcatV2.sequential_20/dense_40/Tensordot/free:output:0.sequential_20/dense_40/Tensordot/axes:output:05sequential_20/dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_20/dense_40/Tensordot/concat?
&sequential_20/dense_40/Tensordot/stackPack.sequential_20/dense_40/Tensordot/Prod:output:00sequential_20/dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_20/dense_40/Tensordot/stack?
*sequential_20/dense_40/Tensordot/transpose	Transposeinput_210sequential_20/dense_40/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22,
*sequential_20/dense_40/Tensordot/transpose?
(sequential_20/dense_40/Tensordot/ReshapeReshape.sequential_20/dense_40/Tensordot/transpose:y:0/sequential_20/dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_20/dense_40/Tensordot/Reshape?
'sequential_20/dense_40/Tensordot/MatMulMatMul1sequential_20/dense_40/Tensordot/Reshape:output:07sequential_20/dense_40/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_20/dense_40/Tensordot/MatMul?
(sequential_20/dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_20/dense_40/Tensordot/Const_2?
.sequential_20/dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_20/dense_40/Tensordot/concat_1/axis?
)sequential_20/dense_40/Tensordot/concat_1ConcatV22sequential_20/dense_40/Tensordot/GatherV2:output:01sequential_20/dense_40/Tensordot/Const_2:output:07sequential_20/dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_20/dense_40/Tensordot/concat_1?
 sequential_20/dense_40/TensordotReshape1sequential_20/dense_40/Tensordot/MatMul:product:02sequential_20/dense_40/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2"
 sequential_20/dense_40/Tensordot?
-sequential_20/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_20/dense_40/BiasAdd/ReadVariableOp?
sequential_20/dense_40/BiasAddBiasAdd)sequential_20/dense_40/Tensordot:output:05sequential_20/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2 
sequential_20/dense_40/BiasAdd?
sequential_20/dense_40/ReluRelu'sequential_20/dense_40/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
sequential_20/dense_40/Relu?
/sequential_20/dense_41/Tensordot/ReadVariableOpReadVariableOp8sequential_20_dense_41_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_20/dense_41/Tensordot/ReadVariableOp?
%sequential_20/dense_41/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_20/dense_41/Tensordot/axes?
%sequential_20/dense_41/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_20/dense_41/Tensordot/free?
&sequential_20/dense_41/Tensordot/ShapeShape)sequential_20/dense_40/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_20/dense_41/Tensordot/Shape?
.sequential_20/dense_41/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_20/dense_41/Tensordot/GatherV2/axis?
)sequential_20/dense_41/Tensordot/GatherV2GatherV2/sequential_20/dense_41/Tensordot/Shape:output:0.sequential_20/dense_41/Tensordot/free:output:07sequential_20/dense_41/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_20/dense_41/Tensordot/GatherV2?
0sequential_20/dense_41/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_20/dense_41/Tensordot/GatherV2_1/axis?
+sequential_20/dense_41/Tensordot/GatherV2_1GatherV2/sequential_20/dense_41/Tensordot/Shape:output:0.sequential_20/dense_41/Tensordot/axes:output:09sequential_20/dense_41/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_20/dense_41/Tensordot/GatherV2_1?
&sequential_20/dense_41/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_20/dense_41/Tensordot/Const?
%sequential_20/dense_41/Tensordot/ProdProd2sequential_20/dense_41/Tensordot/GatherV2:output:0/sequential_20/dense_41/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_20/dense_41/Tensordot/Prod?
(sequential_20/dense_41/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_20/dense_41/Tensordot/Const_1?
'sequential_20/dense_41/Tensordot/Prod_1Prod4sequential_20/dense_41/Tensordot/GatherV2_1:output:01sequential_20/dense_41/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_20/dense_41/Tensordot/Prod_1?
,sequential_20/dense_41/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_20/dense_41/Tensordot/concat/axis?
'sequential_20/dense_41/Tensordot/concatConcatV2.sequential_20/dense_41/Tensordot/free:output:0.sequential_20/dense_41/Tensordot/axes:output:05sequential_20/dense_41/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_20/dense_41/Tensordot/concat?
&sequential_20/dense_41/Tensordot/stackPack.sequential_20/dense_41/Tensordot/Prod:output:00sequential_20/dense_41/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_20/dense_41/Tensordot/stack?
*sequential_20/dense_41/Tensordot/transpose	Transpose)sequential_20/dense_40/Relu:activations:00sequential_20/dense_41/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2,
*sequential_20/dense_41/Tensordot/transpose?
(sequential_20/dense_41/Tensordot/ReshapeReshape.sequential_20/dense_41/Tensordot/transpose:y:0/sequential_20/dense_41/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_20/dense_41/Tensordot/Reshape?
'sequential_20/dense_41/Tensordot/MatMulMatMul1sequential_20/dense_41/Tensordot/Reshape:output:07sequential_20/dense_41/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_20/dense_41/Tensordot/MatMul?
(sequential_20/dense_41/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_20/dense_41/Tensordot/Const_2?
.sequential_20/dense_41/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_20/dense_41/Tensordot/concat_1/axis?
)sequential_20/dense_41/Tensordot/concat_1ConcatV22sequential_20/dense_41/Tensordot/GatherV2:output:01sequential_20/dense_41/Tensordot/Const_2:output:07sequential_20/dense_41/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_20/dense_41/Tensordot/concat_1?
 sequential_20/dense_41/TensordotReshape1sequential_20/dense_41/Tensordot/MatMul:product:02sequential_20/dense_41/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22"
 sequential_20/dense_41/Tensordot?
-sequential_20/dense_41/BiasAdd/ReadVariableOpReadVariableOp6sequential_20_dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_20/dense_41/BiasAdd/ReadVariableOp?
sequential_20/dense_41/BiasAddBiasAdd)sequential_20/dense_41/Tensordot:output:05sequential_20/dense_41/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22 
sequential_20/dense_41/BiasAdd
IdentityIdentity'sequential_20/dense_41/BiasAdd:output:0*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2:::::U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_21
?
?
E__inference_dense_40_layer_call_and_return_conditional_losses_3659492

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????2:::S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659569
input_21
dense_40_3659558
dense_40_3659560
dense_41_3659563
dense_41_3659565
identity?? dense_40/StatefulPartitionedCall? dense_41/StatefulPartitionedCall?
 dense_40/StatefulPartitionedCallStatefulPartitionedCallinput_21dense_40_3659558dense_40_3659560*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2 *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_40_layer_call_and_return_conditional_losses_36594922"
 dense_40/StatefulPartitionedCall?
 dense_41/StatefulPartitionedCallStatefulPartitionedCall)dense_40/StatefulPartitionedCall:output:0dense_41_3659563dense_41_3659565*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_36595382"
 dense_41/StatefulPartitionedCall?
IdentityIdentity)dense_41/StatefulPartitionedCall:output:0!^dense_40/StatefulPartitionedCall!^dense_41/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall2D
 dense_41/StatefulPartitionedCall dense_41/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_21
?
?
E__inference_dense_40_layer_call_and_return_conditional_losses_3659810

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2	
BiasAdd\
ReluReluBiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
Reluj
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:?????????2 2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????2:::S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?

*__inference_dense_41_layer_call_fn_3659858

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
 *+
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_41_layer_call_and_return_conditional_losses_36595382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????2 ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?
?
E__inference_dense_41_layer_call_and_return_conditional_losses_3659849

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity??
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22	
BiasAddh
IdentityIdentityBiasAdd:output:0*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????2 :::S O
+
_output_shapes
:?????????2 
 
_user_specified_nameinputs
?E
?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659696

inputs.
*dense_40_tensordot_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource.
*dense_41_tensordot_readvariableop_resource,
(dense_41_biasadd_readvariableop_resource
identity??
!dense_40/Tensordot/ReadVariableOpReadVariableOp*dense_40_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_40/Tensordot/ReadVariableOp|
dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_40/Tensordot/axes?
dense_40/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_40/Tensordot/freej
dense_40/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_40/Tensordot/Shape?
 dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_40/Tensordot/GatherV2/axis?
dense_40/Tensordot/GatherV2GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/free:output:0)dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_40/Tensordot/GatherV2?
"dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_40/Tensordot/GatherV2_1/axis?
dense_40/Tensordot/GatherV2_1GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/axes:output:0+dense_40/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_40/Tensordot/GatherV2_1~
dense_40/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_40/Tensordot/Const?
dense_40/Tensordot/ProdProd$dense_40/Tensordot/GatherV2:output:0!dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_40/Tensordot/Prod?
dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_40/Tensordot/Const_1?
dense_40/Tensordot/Prod_1Prod&dense_40/Tensordot/GatherV2_1:output:0#dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_40/Tensordot/Prod_1?
dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_40/Tensordot/concat/axis?
dense_40/Tensordot/concatConcatV2 dense_40/Tensordot/free:output:0 dense_40/Tensordot/axes:output:0'dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/concat?
dense_40/Tensordot/stackPack dense_40/Tensordot/Prod:output:0"dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/stack?
dense_40/Tensordot/transpose	Transposeinputs"dense_40/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_40/Tensordot/transpose?
dense_40/Tensordot/ReshapeReshape dense_40/Tensordot/transpose:y:0!dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_40/Tensordot/Reshape?
dense_40/Tensordot/MatMulMatMul#dense_40/Tensordot/Reshape:output:0)dense_40/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_40/Tensordot/MatMul?
dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_40/Tensordot/Const_2?
 dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_40/Tensordot/concat_1/axis?
dense_40/Tensordot/concat_1ConcatV2$dense_40/Tensordot/GatherV2:output:0#dense_40/Tensordot/Const_2:output:0)dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/concat_1?
dense_40/TensordotReshape#dense_40/Tensordot/MatMul:product:0$dense_40/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_40/Tensordot?
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_40/BiasAdd/ReadVariableOp?
dense_40/BiasAddBiasAdddense_40/Tensordot:output:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_40/BiasAddw
dense_40/ReluReludense_40/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_40/Relu?
!dense_41/Tensordot/ReadVariableOpReadVariableOp*dense_41_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_41/Tensordot/ReadVariableOp|
dense_41/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_41/Tensordot/axes?
dense_41/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_41/Tensordot/free
dense_41/Tensordot/ShapeShapedense_40/Relu:activations:0*
T0*
_output_shapes
:2
dense_41/Tensordot/Shape?
 dense_41/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_41/Tensordot/GatherV2/axis?
dense_41/Tensordot/GatherV2GatherV2!dense_41/Tensordot/Shape:output:0 dense_41/Tensordot/free:output:0)dense_41/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_41/Tensordot/GatherV2?
"dense_41/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_41/Tensordot/GatherV2_1/axis?
dense_41/Tensordot/GatherV2_1GatherV2!dense_41/Tensordot/Shape:output:0 dense_41/Tensordot/axes:output:0+dense_41/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_41/Tensordot/GatherV2_1~
dense_41/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_41/Tensordot/Const?
dense_41/Tensordot/ProdProd$dense_41/Tensordot/GatherV2:output:0!dense_41/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_41/Tensordot/Prod?
dense_41/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_41/Tensordot/Const_1?
dense_41/Tensordot/Prod_1Prod&dense_41/Tensordot/GatherV2_1:output:0#dense_41/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_41/Tensordot/Prod_1?
dense_41/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_41/Tensordot/concat/axis?
dense_41/Tensordot/concatConcatV2 dense_41/Tensordot/free:output:0 dense_41/Tensordot/axes:output:0'dense_41/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_41/Tensordot/concat?
dense_41/Tensordot/stackPack dense_41/Tensordot/Prod:output:0"dense_41/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_41/Tensordot/stack?
dense_41/Tensordot/transpose	Transposedense_40/Relu:activations:0"dense_41/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_41/Tensordot/transpose?
dense_41/Tensordot/ReshapeReshape dense_41/Tensordot/transpose:y:0!dense_41/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_41/Tensordot/Reshape?
dense_41/Tensordot/MatMulMatMul#dense_41/Tensordot/Reshape:output:0)dense_41/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_41/Tensordot/MatMul?
dense_41/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_41/Tensordot/Const_2?
 dense_41/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_41/Tensordot/concat_1/axis?
dense_41/Tensordot/concat_1ConcatV2$dense_41/Tensordot/GatherV2:output:0#dense_41/Tensordot/Const_2:output:0)dense_41/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_41/Tensordot/concat_1?
dense_41/TensordotReshape#dense_41/Tensordot/MatMul:product:0$dense_41/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_41/Tensordot?
dense_41/BiasAdd/ReadVariableOpReadVariableOp(dense_41_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_41/BiasAdd/ReadVariableOp?
dense_41/BiasAddBiasAdddense_41/Tensordot:output:0'dense_41/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_41/BiasAddq
IdentityIdentitydense_41/BiasAdd:output:0*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2:::::S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input_215
serving_default_input_21:0?????????2@
dense_414
StatefulPartitionedCall:0?????????2tensorflow/serving/predict:?\
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
$_default_save_signature
%__call__
*&&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_20", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_21"}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_20", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_21"}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
'__call__
*(&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
)__call__
**&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_41", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 32]}}
"
	optimizer
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
non_trainable_variables
	variables

layers
regularization_losses
metrics
layer_metrics
layer_regularization_losses
%__call__
$_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
,
+serving_default"
signature_map
!: 2dense_40/kernel
: 2dense_40/bias
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
non_trainable_variables
	variables

layers
regularization_losses
metrics
layer_metrics
layer_regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
!: 2dense_41/kernel
:2dense_41/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
non_trainable_variables
	variables

 layers
regularization_losses
!metrics
"layer_metrics
#layer_regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
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
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?2?
"__inference__wrapped_model_3659457?
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
annotations? *+?(
&?#
input_21?????????2
?2?
/__inference_sequential_20_layer_call_fn_3659597
/__inference_sequential_20_layer_call_fn_3659779
/__inference_sequential_20_layer_call_fn_3659766
/__inference_sequential_20_layer_call_fn_3659624?
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
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659753
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659555
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659569
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659696?
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
*__inference_dense_40_layer_call_fn_3659819?
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
E__inference_dense_40_layer_call_and_return_conditional_losses_3659810?
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
*__inference_dense_41_layer_call_fn_3659858?
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
E__inference_dense_41_layer_call_and_return_conditional_losses_3659849?
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
%__inference_signature_wrapper_3659639input_21?
"__inference__wrapped_model_3659457v	
5?2
+?(
&?#
input_21?????????2
? "7?4
2
dense_41&?#
dense_41?????????2?
E__inference_dense_40_layer_call_and_return_conditional_losses_3659810d	
3?0
)?&
$?!
inputs?????????2
? ")?&
?
0?????????2 
? ?
*__inference_dense_40_layer_call_fn_3659819W	
3?0
)?&
$?!
inputs?????????2
? "??????????2 ?
E__inference_dense_41_layer_call_and_return_conditional_losses_3659849d3?0
)?&
$?!
inputs?????????2 
? ")?&
?
0?????????2
? ?
*__inference_dense_41_layer_call_fn_3659858W3?0
)?&
$?!
inputs?????????2 
? "??????????2?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659555p	
=?:
3?0
&?#
input_21?????????2
p

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659569p	
=?:
3?0
&?#
input_21?????????2
p 

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659696n	
;?8
1?.
$?!
inputs?????????2
p

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_20_layer_call_and_return_conditional_losses_3659753n	
;?8
1?.
$?!
inputs?????????2
p 

 
? ")?&
?
0?????????2
? ?
/__inference_sequential_20_layer_call_fn_3659597c	
=?:
3?0
&?#
input_21?????????2
p

 
? "??????????2?
/__inference_sequential_20_layer_call_fn_3659624c	
=?:
3?0
&?#
input_21?????????2
p 

 
? "??????????2?
/__inference_sequential_20_layer_call_fn_3659766a	
;?8
1?.
$?!
inputs?????????2
p

 
? "??????????2?
/__inference_sequential_20_layer_call_fn_3659779a	
;?8
1?.
$?!
inputs?????????2
p 

 
? "??????????2?
%__inference_signature_wrapper_3659639?	
A?>
? 
7?4
2
input_21&?#
input_21?????????2"7?4
2
dense_41&?#
dense_41?????????2