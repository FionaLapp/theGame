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
dense_86/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
 * 
shared_namedense_86/kernel
s
#dense_86/kernel/Read/ReadVariableOpReadVariableOpdense_86/kernel*
_output_shapes

:
 *
dtype0
r
dense_86/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_86/bias
k
!dense_86/bias/Read/ReadVariableOpReadVariableOpdense_86/bias*
_output_shapes
: *
dtype0
z
dense_87/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_87/kernel
s
#dense_87/kernel/Read/ReadVariableOpReadVariableOpdense_87/kernel*
_output_shapes

: *
dtype0
r
dense_87/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_87/bias
k
!dense_87/bias/Read/ReadVariableOpReadVariableOpdense_87/bias*
_output_shapes
:*
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
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

	kernel

bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 
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
?
regularization_losses
metrics
non_trainable_variables
	variables
layer_regularization_losses
layer_metrics
trainable_variables

layers
 
[Y
VARIABLE_VALUEdense_86/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_86/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
?
regularization_losses
metrics
non_trainable_variables
layer_regularization_losses
	variables
layer_metrics
trainable_variables

layers
[Y
VARIABLE_VALUEdense_87/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_87/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
metrics
 non_trainable_variables
!layer_regularization_losses
	variables
"layer_metrics
trainable_variables

#layers
 
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
{
serving_default_input_37Placeholder*'
_output_shapes
:?????????
*
dtype0*
shape:?????????

?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_37dense_86/kerneldense_86/biasdense_87/kerneldense_87/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_12532
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_86/kernel/Read/ReadVariableOp!dense_86/bias/Read/ReadVariableOp#dense_87/kernel/Read/ReadVariableOp!dense_87/bias/Read/ReadVariableOpConst*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_12666
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_86/kerneldense_86/biasdense_87/kerneldense_87/bias*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_12688??
?
?
C__inference_dense_86_layer_call_and_return_conditional_losses_12405

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
 *
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
:?????????
:::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_sequential_40_layer_call_and_return_conditional_losses_12549

inputs+
'dense_86_matmul_readvariableop_resource,
(dense_86_biasadd_readvariableop_resource+
'dense_87_matmul_readvariableop_resource,
(dense_87_biasadd_readvariableop_resource
identity??
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype02 
dense_86/MatMul/ReadVariableOp?
dense_86/MatMulMatMulinputs&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_86/MatMul?
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_86/BiasAdd/ReadVariableOp?
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_86/BiasAdds
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_86/Relu?
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_87/MatMul/ReadVariableOp?
dense_87/MatMulMatMuldense_86/Relu:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_87/MatMul?
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_87/BiasAdd/ReadVariableOp?
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_87/BiasAddm
IdentityIdentitydense_87/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
:::::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_sequential_40_layer_call_and_return_conditional_losses_12448
input_37
dense_86_12416
dense_86_12418
dense_87_12442
dense_87_12444
identity?? dense_86/StatefulPartitionedCall? dense_87/StatefulPartitionedCall?
 dense_86/StatefulPartitionedCallStatefulPartitionedCallinput_37dense_86_12416dense_86_12418*
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
GPU 2J 8? *L
fGRE
C__inference_dense_86_layer_call_and_return_conditional_losses_124052"
 dense_86/StatefulPartitionedCall?
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_12442dense_87_12444*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_87_layer_call_and_return_conditional_losses_124312"
 dense_87/StatefulPartitionedCall?
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
input_37
?
?
-__inference_sequential_40_layer_call_fn_12592

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
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_40_layer_call_and_return_conditional_losses_125062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_sequential_40_layer_call_and_return_conditional_losses_12506

inputs
dense_86_12495
dense_86_12497
dense_87_12500
dense_87_12502
identity?? dense_86/StatefulPartitionedCall? dense_87/StatefulPartitionedCall?
 dense_86/StatefulPartitionedCallStatefulPartitionedCallinputsdense_86_12495dense_86_12497*
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
GPU 2J 8? *L
fGRE
C__inference_dense_86_layer_call_and_return_conditional_losses_124052"
 dense_86/StatefulPartitionedCall?
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_12500dense_87_12502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_87_layer_call_and_return_conditional_losses_124312"
 dense_87/StatefulPartitionedCall?
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
 __inference__wrapped_model_12390
input_379
5sequential_40_dense_86_matmul_readvariableop_resource:
6sequential_40_dense_86_biasadd_readvariableop_resource9
5sequential_40_dense_87_matmul_readvariableop_resource:
6sequential_40_dense_87_biasadd_readvariableop_resource
identity??
,sequential_40/dense_86/MatMul/ReadVariableOpReadVariableOp5sequential_40_dense_86_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype02.
,sequential_40/dense_86/MatMul/ReadVariableOp?
sequential_40/dense_86/MatMulMatMulinput_374sequential_40/dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
sequential_40/dense_86/MatMul?
-sequential_40/dense_86/BiasAdd/ReadVariableOpReadVariableOp6sequential_40_dense_86_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_40/dense_86/BiasAdd/ReadVariableOp?
sequential_40/dense_86/BiasAddBiasAdd'sequential_40/dense_86/MatMul:product:05sequential_40/dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2 
sequential_40/dense_86/BiasAdd?
sequential_40/dense_86/ReluRelu'sequential_40/dense_86/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
sequential_40/dense_86/Relu?
,sequential_40/dense_87/MatMul/ReadVariableOpReadVariableOp5sequential_40_dense_87_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,sequential_40/dense_87/MatMul/ReadVariableOp?
sequential_40/dense_87/MatMulMatMul)sequential_40/dense_86/Relu:activations:04sequential_40/dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential_40/dense_87/MatMul?
-sequential_40/dense_87/BiasAdd/ReadVariableOpReadVariableOp6sequential_40_dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_40/dense_87/BiasAdd/ReadVariableOp?
sequential_40/dense_87/BiasAddBiasAdd'sequential_40/dense_87/MatMul:product:05sequential_40/dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential_40/dense_87/BiasAdd{
IdentityIdentity'sequential_40/dense_87/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
:::::Q M
'
_output_shapes
:?????????

"
_user_specified_name
input_37
?
?
#__inference_signature_wrapper_12532
input_37
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_37unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_123902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
input_37
?
?
__inference__traced_save_12666
file_prefix.
*savev2_dense_86_kernel_read_readvariableop,
(savev2_dense_86_bias_read_readvariableop.
*savev2_dense_87_kernel_read_readvariableop,
(savev2_dense_87_bias_read_readvariableop
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
value3B1 B+_temp_62355d16120b48899fb02c69897b7053/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_86_kernel_read_readvariableop(savev2_dense_86_bias_read_readvariableop*savev2_dense_87_kernel_read_readvariableop(savev2_dense_87_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
$: :
 : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:
 : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?
?
H__inference_sequential_40_layer_call_and_return_conditional_losses_12462
input_37
dense_86_12451
dense_86_12453
dense_87_12456
dense_87_12458
identity?? dense_86/StatefulPartitionedCall? dense_87/StatefulPartitionedCall?
 dense_86/StatefulPartitionedCallStatefulPartitionedCallinput_37dense_86_12451dense_86_12453*
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
GPU 2J 8? *L
fGRE
C__inference_dense_86_layer_call_and_return_conditional_losses_124052"
 dense_86/StatefulPartitionedCall?
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_12456dense_87_12458*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_87_layer_call_and_return_conditional_losses_124312"
 dense_87/StatefulPartitionedCall?
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
input_37
?
?
-__inference_sequential_40_layer_call_fn_12490
input_37
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_37unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_40_layer_call_and_return_conditional_losses_124792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
input_37
?
?
-__inference_sequential_40_layer_call_fn_12579

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
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_40_layer_call_and_return_conditional_losses_124792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
H__inference_sequential_40_layer_call_and_return_conditional_losses_12479

inputs
dense_86_12468
dense_86_12470
dense_87_12473
dense_87_12475
identity?? dense_86/StatefulPartitionedCall? dense_87/StatefulPartitionedCall?
 dense_86/StatefulPartitionedCallStatefulPartitionedCallinputsdense_86_12468dense_86_12470*
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
GPU 2J 8? *L
fGRE
C__inference_dense_86_layer_call_and_return_conditional_losses_124052"
 dense_86/StatefulPartitionedCall?
 dense_87/StatefulPartitionedCallStatefulPartitionedCall)dense_86/StatefulPartitionedCall:output:0dense_87_12473dense_87_12475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_87_layer_call_and_return_conditional_losses_124312"
 dense_87/StatefulPartitionedCall?
IdentityIdentity)dense_87/StatefulPartitionedCall:output:0!^dense_86/StatefulPartitionedCall!^dense_87/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::2D
 dense_86/StatefulPartitionedCall dense_86/StatefulPartitionedCall2D
 dense_87/StatefulPartitionedCall dense_87/StatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
!__inference__traced_restore_12688
file_prefix$
 assignvariableop_dense_86_kernel$
 assignvariableop_1_dense_86_bias&
"assignvariableop_2_dense_87_kernel$
 assignvariableop_3_dense_87_bias

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
AssignVariableOpAssignVariableOp assignvariableop_dense_86_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_86_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_87_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_87_biasIdentity_3:output:0"/device:CPU:0*
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
?
}
(__inference_dense_86_layer_call_fn_12612

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
GPU 2J 8? *L
fGRE
C__inference_dense_86_layer_call_and_return_conditional_losses_124052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
C__inference_dense_87_layer_call_and_return_conditional_losses_12431

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

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
H__inference_sequential_40_layer_call_and_return_conditional_losses_12566

inputs+
'dense_86_matmul_readvariableop_resource,
(dense_86_biasadd_readvariableop_resource+
'dense_87_matmul_readvariableop_resource,
(dense_87_biasadd_readvariableop_resource
identity??
dense_86/MatMul/ReadVariableOpReadVariableOp'dense_86_matmul_readvariableop_resource*
_output_shapes

:
 *
dtype02 
dense_86/MatMul/ReadVariableOp?
dense_86/MatMulMatMulinputs&dense_86/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_86/MatMul?
dense_86/BiasAdd/ReadVariableOpReadVariableOp(dense_86_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_86/BiasAdd/ReadVariableOp?
dense_86/BiasAddBiasAdddense_86/MatMul:product:0'dense_86/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_86/BiasAdds
dense_86/ReluReludense_86/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense_86/Relu?
dense_87/MatMul/ReadVariableOpReadVariableOp'dense_87_matmul_readvariableop_resource*
_output_shapes

: *
dtype02 
dense_87/MatMul/ReadVariableOp?
dense_87/MatMulMatMuldense_86/Relu:activations:0&dense_87/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_87/MatMul?
dense_87/BiasAdd/ReadVariableOpReadVariableOp(dense_87_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_87/BiasAdd/ReadVariableOp?
dense_87/BiasAddBiasAdddense_87/MatMul:product:0'dense_87/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_87/BiasAddm
IdentityIdentitydense_87/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
:::::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
-__inference_sequential_40_layer_call_fn_12517
input_37
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_37unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_sequential_40_layer_call_and_return_conditional_losses_125062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????
::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????

"
_user_specified_name
input_37
?
?
C__inference_dense_87_layer_call_and_return_conditional_losses_12622

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :::O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
C__inference_dense_86_layer_call_and_return_conditional_losses_12603

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
 *
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
:?????????
:::O K
'
_output_shapes
:?????????

 
_user_specified_nameinputs
?
}
(__inference_dense_87_layer_call_fn_12631

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
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_dense_87_layer_call_and_return_conditional_losses_124312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
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
input_371
serving_default_input_37:0?????????
<
dense_870
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?[
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
$_default_save_signature
%__call__
*&&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_40", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_40", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_37"}}, {"class_name": "Dense", "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_40", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 10]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_37"}}, {"class_name": "Dense", "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
?

	kernel

bias
regularization_losses
	variables
trainable_variables
	keras_api
'__call__
*(&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_86", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_86", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
)__call__
**&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_87", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_87", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
"
	optimizer
 "
trackable_list_wrapper
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
?
regularization_losses
metrics
non_trainable_variables
	variables
layer_regularization_losses
layer_metrics
trainable_variables

layers
%__call__
$_default_save_signature
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
,
+serving_default"
signature_map
!:
 2dense_86/kernel
: 2dense_86/bias
 "
trackable_list_wrapper
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
?
regularization_losses
metrics
non_trainable_variables
layer_regularization_losses
	variables
layer_metrics
trainable_variables

layers
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
!: 2dense_87/kernel
:2dense_87/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
metrics
 non_trainable_variables
!layer_regularization_losses
	variables
"layer_metrics
trainable_variables

#layers
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
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
 __inference__wrapped_model_12390?
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
input_37?????????

?2?
-__inference_sequential_40_layer_call_fn_12592
-__inference_sequential_40_layer_call_fn_12579
-__inference_sequential_40_layer_call_fn_12490
-__inference_sequential_40_layer_call_fn_12517?
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
H__inference_sequential_40_layer_call_and_return_conditional_losses_12448
H__inference_sequential_40_layer_call_and_return_conditional_losses_12566
H__inference_sequential_40_layer_call_and_return_conditional_losses_12462
H__inference_sequential_40_layer_call_and_return_conditional_losses_12549?
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
(__inference_dense_86_layer_call_fn_12612?
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
C__inference_dense_86_layer_call_and_return_conditional_losses_12603?
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
(__inference_dense_87_layer_call_fn_12631?
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
C__inference_dense_87_layer_call_and_return_conditional_losses_12622?
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
3B1
#__inference_signature_wrapper_12532input_37?
 __inference__wrapped_model_12390n	
1?.
'?$
"?
input_37?????????

? "3?0
.
dense_87"?
dense_87??????????
C__inference_dense_86_layer_call_and_return_conditional_losses_12603\	
/?,
%?"
 ?
inputs?????????

? "%?"
?
0????????? 
? {
(__inference_dense_86_layer_call_fn_12612O	
/?,
%?"
 ?
inputs?????????

? "?????????? ?
C__inference_dense_87_layer_call_and_return_conditional_losses_12622\/?,
%?"
 ?
inputs????????? 
? "%?"
?
0?????????
? {
(__inference_dense_87_layer_call_fn_12631O/?,
%?"
 ?
inputs????????? 
? "???????????
H__inference_sequential_40_layer_call_and_return_conditional_losses_12448h	
9?6
/?,
"?
input_37?????????

p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_40_layer_call_and_return_conditional_losses_12462h	
9?6
/?,
"?
input_37?????????

p 

 
? "%?"
?
0?????????
? ?
H__inference_sequential_40_layer_call_and_return_conditional_losses_12549f	
7?4
-?*
 ?
inputs?????????

p

 
? "%?"
?
0?????????
? ?
H__inference_sequential_40_layer_call_and_return_conditional_losses_12566f	
7?4
-?*
 ?
inputs?????????

p 

 
? "%?"
?
0?????????
? ?
-__inference_sequential_40_layer_call_fn_12490[	
9?6
/?,
"?
input_37?????????

p

 
? "???????????
-__inference_sequential_40_layer_call_fn_12517[	
9?6
/?,
"?
input_37?????????

p 

 
? "???????????
-__inference_sequential_40_layer_call_fn_12579Y	
7?4
-?*
 ?
inputs?????????

p

 
? "???????????
-__inference_sequential_40_layer_call_fn_12592Y	
7?4
-?*
 ?
inputs?????????

p 

 
? "???????????
#__inference_signature_wrapper_12532z	
=?:
? 
3?0
.
input_37"?
input_37?????????
"3?0
.
dense_87"?
dense_87?????????