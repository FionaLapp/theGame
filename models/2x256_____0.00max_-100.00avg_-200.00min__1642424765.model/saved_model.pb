
¿£
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
dtypetype
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8²
z
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_50/kernel
s
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel*
_output_shapes

: *
dtype0
r
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_50/bias
k
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes
: *
dtype0
z
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_51/kernel
s
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel*
_output_shapes

:  *
dtype0
r
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_51/bias
k
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes
: *
dtype0
z
dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_52/kernel
s
#dense_52/kernel/Read/ReadVariableOpReadVariableOpdense_52/kernel*
_output_shapes

: *
dtype0
r
dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_52/bias
k
!dense_52/bias/Read/ReadVariableOpReadVariableOpdense_52/bias*
_output_shapes
:*
dtype0

NoOpNoOp
ã
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
æ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
h


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 
 
*

0
1
2
3
4
5
*

0
1
2
3
4
5
­
regularization_losses
metrics
non_trainable_variables
	variables
layer_regularization_losses
layer_metrics
trainable_variables

 layers
 
[Y
VARIABLE_VALUEdense_50/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_50/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 


0
1


0
1
­
regularization_losses
!metrics
"non_trainable_variables
#layer_regularization_losses
	variables
$layer_metrics
trainable_variables

%layers
[Y
VARIABLE_VALUEdense_51/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_51/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
&metrics
'non_trainable_variables
(layer_regularization_losses
	variables
)layer_metrics
trainable_variables

*layers
[Y
VARIABLE_VALUEdense_52/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_52/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
+metrics
,non_trainable_variables
-layer_regularization_losses
	variables
.layer_metrics
trainable_variables

/layers
 
 
 
 

0
1
2
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
 
 
Ã
serving_default_input_19Placeholder*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*
dtype0*@
shape7:5ÿÿÿÿÿÿÿÿÿ
Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19dense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_7755
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOp#dense_52/kernel/Read/ReadVariableOp!dense_52/bias/Read/ReadVariableOpConst*
Tin

2*
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
GPU 2J 8 *&
f!R
__inference__traced_save_8117
ú
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_50/kerneldense_50/biasdense_51/kerneldense_51/biasdense_52/kerneldense_52/bias*
Tin
	2*
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
GPU 2J 8 *)
f$R"
 __inference__traced_restore_8145ëë
ù
¿
,__inference_sequential_22_layer_call_fn_7736
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_22_layer_call_and_return_conditional_losses_77212
StatefulPartitionedCall²
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_19
ù

__inference__traced_save_8117
file_prefix.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop.
*savev2_dense_52_kernel_read_readvariableop,
(savev2_dense_52_bias_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4204bb08804b46028abe966c857840d1/part2	
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameë
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ý
valueóBðB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slicesÂ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop*savev2_dense_52_kernel_read_readvariableop(savev2_dense_52_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*G
_input_shapes6
4: : : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
ó
½
,__inference_sequential_22_layer_call_fn_7940

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_22_layer_call_and_return_conditional_losses_76852
StatefulPartitionedCall²
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ª
 __inference__traced_restore_8145
file_prefix$
 assignvariableop_dense_50_kernel$
 assignvariableop_1_dense_50_bias&
"assignvariableop_2_dense_51_kernel$
 assignvariableop_3_dense_51_bias&
"assignvariableop_4_dense_52_kernel$
 assignvariableop_5_dense_52_bias

identity_7¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5ñ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ý
valueóBðB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slicesÎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_50_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_50_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_51_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_51_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_52_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_52_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpä

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6Ö

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ø
®
__inference__wrapped_model_7499
input_19<
8sequential_22_dense_50_tensordot_readvariableop_resource:
6sequential_22_dense_50_biasadd_readvariableop_resource<
8sequential_22_dense_51_tensordot_readvariableop_resource:
6sequential_22_dense_51_biasadd_readvariableop_resource<
8sequential_22_dense_52_tensordot_readvariableop_resource:
6sequential_22_dense_52_biasadd_readvariableop_resource
identityÛ
/sequential_22/dense_50/Tensordot/ReadVariableOpReadVariableOp8sequential_22_dense_50_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_22/dense_50/Tensordot/ReadVariableOp
%sequential_22/dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_22/dense_50/Tensordot/axes¿
%sequential_22/dense_50/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_22/dense_50/Tensordot/free
&sequential_22/dense_50/Tensordot/ShapeShapeinput_19*
T0*
_output_shapes
:2(
&sequential_22/dense_50/Tensordot/Shape¢
.sequential_22/dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_22/dense_50/Tensordot/GatherV2/axisÄ
)sequential_22/dense_50/Tensordot/GatherV2GatherV2/sequential_22/dense_50/Tensordot/Shape:output:0.sequential_22/dense_50/Tensordot/free:output:07sequential_22/dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_22/dense_50/Tensordot/GatherV2¦
0sequential_22/dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_22/dense_50/Tensordot/GatherV2_1/axisÊ
+sequential_22/dense_50/Tensordot/GatherV2_1GatherV2/sequential_22/dense_50/Tensordot/Shape:output:0.sequential_22/dense_50/Tensordot/axes:output:09sequential_22/dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_22/dense_50/Tensordot/GatherV2_1
&sequential_22/dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_22/dense_50/Tensordot/ConstÜ
%sequential_22/dense_50/Tensordot/ProdProd2sequential_22/dense_50/Tensordot/GatherV2:output:0/sequential_22/dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_22/dense_50/Tensordot/Prod
(sequential_22/dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_22/dense_50/Tensordot/Const_1ä
'sequential_22/dense_50/Tensordot/Prod_1Prod4sequential_22/dense_50/Tensordot/GatherV2_1:output:01sequential_22/dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_22/dense_50/Tensordot/Prod_1
,sequential_22/dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_22/dense_50/Tensordot/concat/axis£
'sequential_22/dense_50/Tensordot/concatConcatV2.sequential_22/dense_50/Tensordot/free:output:0.sequential_22/dense_50/Tensordot/axes:output:05sequential_22/dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_22/dense_50/Tensordot/concatè
&sequential_22/dense_50/Tensordot/stackPack.sequential_22/dense_50/Tensordot/Prod:output:00sequential_22/dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_22/dense_50/Tensordot/stack÷
*sequential_22/dense_50/Tensordot/transpose	Transposeinput_190sequential_22/dense_50/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2,
*sequential_22/dense_50/Tensordot/transposeû
(sequential_22/dense_50/Tensordot/ReshapeReshape.sequential_22/dense_50/Tensordot/transpose:y:0/sequential_22/dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_22/dense_50/Tensordot/Reshapeú
'sequential_22/dense_50/Tensordot/MatMulMatMul1sequential_22/dense_50/Tensordot/Reshape:output:07sequential_22/dense_50/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_22/dense_50/Tensordot/MatMul
(sequential_22/dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_22/dense_50/Tensordot/Const_2¢
.sequential_22/dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_22/dense_50/Tensordot/concat_1/axis°
)sequential_22/dense_50/Tensordot/concat_1ConcatV22sequential_22/dense_50/Tensordot/GatherV2:output:01sequential_22/dense_50/Tensordot/Const_2:output:07sequential_22/dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_22/dense_50/Tensordot/concat_1
 sequential_22/dense_50/TensordotReshape1sequential_22/dense_50/Tensordot/MatMul:product:02sequential_22/dense_50/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2"
 sequential_22/dense_50/TensordotÑ
-sequential_22/dense_50/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_22/dense_50/BiasAdd/ReadVariableOp
sequential_22/dense_50/BiasAddBiasAdd)sequential_22/dense_50/Tensordot:output:05sequential_22/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2 
sequential_22/dense_50/BiasAddÁ
sequential_22/dense_50/ReluRelu'sequential_22/dense_50/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
sequential_22/dense_50/ReluÛ
/sequential_22/dense_51/Tensordot/ReadVariableOpReadVariableOp8sequential_22_dense_51_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype021
/sequential_22/dense_51/Tensordot/ReadVariableOp
%sequential_22/dense_51/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_22/dense_51/Tensordot/axes¿
%sequential_22/dense_51/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_22/dense_51/Tensordot/free©
&sequential_22/dense_51/Tensordot/ShapeShape)sequential_22/dense_50/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_22/dense_51/Tensordot/Shape¢
.sequential_22/dense_51/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_22/dense_51/Tensordot/GatherV2/axisÄ
)sequential_22/dense_51/Tensordot/GatherV2GatherV2/sequential_22/dense_51/Tensordot/Shape:output:0.sequential_22/dense_51/Tensordot/free:output:07sequential_22/dense_51/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_22/dense_51/Tensordot/GatherV2¦
0sequential_22/dense_51/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_22/dense_51/Tensordot/GatherV2_1/axisÊ
+sequential_22/dense_51/Tensordot/GatherV2_1GatherV2/sequential_22/dense_51/Tensordot/Shape:output:0.sequential_22/dense_51/Tensordot/axes:output:09sequential_22/dense_51/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_22/dense_51/Tensordot/GatherV2_1
&sequential_22/dense_51/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_22/dense_51/Tensordot/ConstÜ
%sequential_22/dense_51/Tensordot/ProdProd2sequential_22/dense_51/Tensordot/GatherV2:output:0/sequential_22/dense_51/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_22/dense_51/Tensordot/Prod
(sequential_22/dense_51/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_22/dense_51/Tensordot/Const_1ä
'sequential_22/dense_51/Tensordot/Prod_1Prod4sequential_22/dense_51/Tensordot/GatherV2_1:output:01sequential_22/dense_51/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_22/dense_51/Tensordot/Prod_1
,sequential_22/dense_51/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_22/dense_51/Tensordot/concat/axis£
'sequential_22/dense_51/Tensordot/concatConcatV2.sequential_22/dense_51/Tensordot/free:output:0.sequential_22/dense_51/Tensordot/axes:output:05sequential_22/dense_51/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_22/dense_51/Tensordot/concatè
&sequential_22/dense_51/Tensordot/stackPack.sequential_22/dense_51/Tensordot/Prod:output:00sequential_22/dense_51/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_22/dense_51/Tensordot/stack
*sequential_22/dense_51/Tensordot/transpose	Transpose)sequential_22/dense_50/Relu:activations:00sequential_22/dense_51/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2,
*sequential_22/dense_51/Tensordot/transposeû
(sequential_22/dense_51/Tensordot/ReshapeReshape.sequential_22/dense_51/Tensordot/transpose:y:0/sequential_22/dense_51/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_22/dense_51/Tensordot/Reshapeú
'sequential_22/dense_51/Tensordot/MatMulMatMul1sequential_22/dense_51/Tensordot/Reshape:output:07sequential_22/dense_51/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_22/dense_51/Tensordot/MatMul
(sequential_22/dense_51/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_22/dense_51/Tensordot/Const_2¢
.sequential_22/dense_51/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_22/dense_51/Tensordot/concat_1/axis°
)sequential_22/dense_51/Tensordot/concat_1ConcatV22sequential_22/dense_51/Tensordot/GatherV2:output:01sequential_22/dense_51/Tensordot/Const_2:output:07sequential_22/dense_51/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_22/dense_51/Tensordot/concat_1
 sequential_22/dense_51/TensordotReshape1sequential_22/dense_51/Tensordot/MatMul:product:02sequential_22/dense_51/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2"
 sequential_22/dense_51/TensordotÑ
-sequential_22/dense_51/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_22/dense_51/BiasAdd/ReadVariableOp
sequential_22/dense_51/BiasAddBiasAdd)sequential_22/dense_51/Tensordot:output:05sequential_22/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2 
sequential_22/dense_51/BiasAddÁ
sequential_22/dense_51/ReluRelu'sequential_22/dense_51/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
sequential_22/dense_51/ReluÛ
/sequential_22/dense_52/Tensordot/ReadVariableOpReadVariableOp8sequential_22_dense_52_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_22/dense_52/Tensordot/ReadVariableOp
%sequential_22/dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_22/dense_52/Tensordot/axes¿
%sequential_22/dense_52/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_22/dense_52/Tensordot/free©
&sequential_22/dense_52/Tensordot/ShapeShape)sequential_22/dense_51/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_22/dense_52/Tensordot/Shape¢
.sequential_22/dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_22/dense_52/Tensordot/GatherV2/axisÄ
)sequential_22/dense_52/Tensordot/GatherV2GatherV2/sequential_22/dense_52/Tensordot/Shape:output:0.sequential_22/dense_52/Tensordot/free:output:07sequential_22/dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_22/dense_52/Tensordot/GatherV2¦
0sequential_22/dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_22/dense_52/Tensordot/GatherV2_1/axisÊ
+sequential_22/dense_52/Tensordot/GatherV2_1GatherV2/sequential_22/dense_52/Tensordot/Shape:output:0.sequential_22/dense_52/Tensordot/axes:output:09sequential_22/dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_22/dense_52/Tensordot/GatherV2_1
&sequential_22/dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_22/dense_52/Tensordot/ConstÜ
%sequential_22/dense_52/Tensordot/ProdProd2sequential_22/dense_52/Tensordot/GatherV2:output:0/sequential_22/dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_22/dense_52/Tensordot/Prod
(sequential_22/dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_22/dense_52/Tensordot/Const_1ä
'sequential_22/dense_52/Tensordot/Prod_1Prod4sequential_22/dense_52/Tensordot/GatherV2_1:output:01sequential_22/dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_22/dense_52/Tensordot/Prod_1
,sequential_22/dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_22/dense_52/Tensordot/concat/axis£
'sequential_22/dense_52/Tensordot/concatConcatV2.sequential_22/dense_52/Tensordot/free:output:0.sequential_22/dense_52/Tensordot/axes:output:05sequential_22/dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_22/dense_52/Tensordot/concatè
&sequential_22/dense_52/Tensordot/stackPack.sequential_22/dense_52/Tensordot/Prod:output:00sequential_22/dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_22/dense_52/Tensordot/stack
*sequential_22/dense_52/Tensordot/transpose	Transpose)sequential_22/dense_51/Relu:activations:00sequential_22/dense_52/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2,
*sequential_22/dense_52/Tensordot/transposeû
(sequential_22/dense_52/Tensordot/ReshapeReshape.sequential_22/dense_52/Tensordot/transpose:y:0/sequential_22/dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_22/dense_52/Tensordot/Reshapeú
'sequential_22/dense_52/Tensordot/MatMulMatMul1sequential_22/dense_52/Tensordot/Reshape:output:07sequential_22/dense_52/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_22/dense_52/Tensordot/MatMul
(sequential_22/dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_22/dense_52/Tensordot/Const_2¢
.sequential_22/dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_22/dense_52/Tensordot/concat_1/axis°
)sequential_22/dense_52/Tensordot/concat_1ConcatV22sequential_22/dense_52/Tensordot/GatherV2:output:01sequential_22/dense_52/Tensordot/Const_2:output:07sequential_22/dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_22/dense_52/Tensordot/concat_1
 sequential_22/dense_52/TensordotReshape1sequential_22/dense_52/Tensordot/MatMul:product:02sequential_22/dense_52/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2"
 sequential_22/dense_52/TensordotÑ
-sequential_22/dense_52/BiasAdd/ReadVariableOpReadVariableOp6sequential_22_dense_52_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_22/dense_52/BiasAdd/ReadVariableOp
sequential_22/dense_52/BiasAddBiasAdd)sequential_22/dense_52/Tensordot:output:05sequential_22/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2 
sequential_22/dense_52/BiasAdd
IdentityIdentity'sequential_22/dense_52/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ:::::::u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_19

Ã
G__inference_sequential_22_layer_call_and_return_conditional_losses_7644
input_19
dense_50_7545
dense_50_7547
dense_51_7592
dense_51_7594
dense_52_7638
dense_52_7640
identity¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall´
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_50_7545dense_50_7547*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_50_layer_call_and_return_conditional_losses_75342"
 dense_50/StatefulPartitionedCallÕ
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_7592dense_51_7594*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_51_layer_call_and_return_conditional_losses_75812"
 dense_51/StatefulPartitionedCallÕ
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_7638dense_52_7640*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_52_layer_call_and_return_conditional_losses_76272"
 dense_52/StatefulPartitionedCall
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_19
à 
­
B__inference_dense_51_layer_call_and_return_conditional_losses_7581

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
Tensordot/axes
Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack°
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1°
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd|
ReluReluBiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ :::s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ù
¿
,__inference_sequential_22_layer_call_fn_7700
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_22_layer_call_and_return_conditional_losses_76852
StatefulPartitionedCall²
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_19
ÿk

G__inference_sequential_22_layer_call_and_return_conditional_losses_7839

inputs.
*dense_50_tensordot_readvariableop_resource,
(dense_50_biasadd_readvariableop_resource.
*dense_51_tensordot_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource.
*dense_52_tensordot_readvariableop_resource,
(dense_52_biasadd_readvariableop_resource
identity±
!dense_50/Tensordot/ReadVariableOpReadVariableOp*dense_50_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_50/Tensordot/ReadVariableOp|
dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_50/Tensordot/axes£
dense_50/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_50/Tensordot/freej
dense_50/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_50/Tensordot/Shape
 dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/GatherV2/axisþ
dense_50/Tensordot/GatherV2GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/free:output:0)dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_50/Tensordot/GatherV2
"dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_50/Tensordot/GatherV2_1/axis
dense_50/Tensordot/GatherV2_1GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/axes:output:0+dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2_1~
dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const¤
dense_50/Tensordot/ProdProd$dense_50/Tensordot/GatherV2:output:0!dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod
dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const_1¬
dense_50/Tensordot/Prod_1Prod&dense_50/Tensordot/GatherV2_1:output:0#dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod_1
dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_50/Tensordot/concat/axisÝ
dense_50/Tensordot/concatConcatV2 dense_50/Tensordot/free:output:0 dense_50/Tensordot/axes:output:0'dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat°
dense_50/Tensordot/stackPack dense_50/Tensordot/Prod:output:0"dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/stackË
dense_50/Tensordot/transpose	Transposeinputs"dense_50/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_50/Tensordot/transposeÃ
dense_50/Tensordot/ReshapeReshape dense_50/Tensordot/transpose:y:0!dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_50/Tensordot/ReshapeÂ
dense_50/Tensordot/MatMulMatMul#dense_50/Tensordot/Reshape:output:0)dense_50/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_50/Tensordot/MatMul
dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const_2
 dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/concat_1/axisê
dense_50/Tensordot/concat_1ConcatV2$dense_50/Tensordot/GatherV2:output:0#dense_50/Tensordot/Const_2:output:0)dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat_1Ô
dense_50/TensordotReshape#dense_50/Tensordot/MatMul:product:0$dense_50/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_50/Tensordot§
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_50/BiasAdd/ReadVariableOpË
dense_50/BiasAddBiasAdddense_50/Tensordot:output:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_50/BiasAdd
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_50/Relu±
!dense_51/Tensordot/ReadVariableOpReadVariableOp*dense_51_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_51/Tensordot/ReadVariableOp|
dense_51/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_51/Tensordot/axes£
dense_51/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_51/Tensordot/free
dense_51/Tensordot/ShapeShapedense_50/Relu:activations:0*
T0*
_output_shapes
:2
dense_51/Tensordot/Shape
 dense_51/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_51/Tensordot/GatherV2/axisþ
dense_51/Tensordot/GatherV2GatherV2!dense_51/Tensordot/Shape:output:0 dense_51/Tensordot/free:output:0)dense_51/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_51/Tensordot/GatherV2
"dense_51/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_51/Tensordot/GatherV2_1/axis
dense_51/Tensordot/GatherV2_1GatherV2!dense_51/Tensordot/Shape:output:0 dense_51/Tensordot/axes:output:0+dense_51/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_51/Tensordot/GatherV2_1~
dense_51/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const¤
dense_51/Tensordot/ProdProd$dense_51/Tensordot/GatherV2:output:0!dense_51/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_51/Tensordot/Prod
dense_51/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const_1¬
dense_51/Tensordot/Prod_1Prod&dense_51/Tensordot/GatherV2_1:output:0#dense_51/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_51/Tensordot/Prod_1
dense_51/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_51/Tensordot/concat/axisÝ
dense_51/Tensordot/concatConcatV2 dense_51/Tensordot/free:output:0 dense_51/Tensordot/axes:output:0'dense_51/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/concat°
dense_51/Tensordot/stackPack dense_51/Tensordot/Prod:output:0"dense_51/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/stackà
dense_51/Tensordot/transpose	Transposedense_50/Relu:activations:0"dense_51/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_51/Tensordot/transposeÃ
dense_51/Tensordot/ReshapeReshape dense_51/Tensordot/transpose:y:0!dense_51/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_51/Tensordot/ReshapeÂ
dense_51/Tensordot/MatMulMatMul#dense_51/Tensordot/Reshape:output:0)dense_51/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_51/Tensordot/MatMul
dense_51/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const_2
 dense_51/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_51/Tensordot/concat_1/axisê
dense_51/Tensordot/concat_1ConcatV2$dense_51/Tensordot/GatherV2:output:0#dense_51/Tensordot/Const_2:output:0)dense_51/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/concat_1Ô
dense_51/TensordotReshape#dense_51/Tensordot/MatMul:product:0$dense_51/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_51/Tensordot§
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_51/BiasAdd/ReadVariableOpË
dense_51/BiasAddBiasAdddense_51/Tensordot:output:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_51/BiasAdd
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_51/Relu±
!dense_52/Tensordot/ReadVariableOpReadVariableOp*dense_52_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_52/Tensordot/ReadVariableOp|
dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_52/Tensordot/axes£
dense_52/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_52/Tensordot/free
dense_52/Tensordot/ShapeShapedense_51/Relu:activations:0*
T0*
_output_shapes
:2
dense_52/Tensordot/Shape
 dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_52/Tensordot/GatherV2/axisþ
dense_52/Tensordot/GatherV2GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/free:output:0)dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_52/Tensordot/GatherV2
"dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_52/Tensordot/GatherV2_1/axis
dense_52/Tensordot/GatherV2_1GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/axes:output:0+dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_52/Tensordot/GatherV2_1~
dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_52/Tensordot/Const¤
dense_52/Tensordot/ProdProd$dense_52/Tensordot/GatherV2:output:0!dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_52/Tensordot/Prod
dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_52/Tensordot/Const_1¬
dense_52/Tensordot/Prod_1Prod&dense_52/Tensordot/GatherV2_1:output:0#dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_52/Tensordot/Prod_1
dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_52/Tensordot/concat/axisÝ
dense_52/Tensordot/concatConcatV2 dense_52/Tensordot/free:output:0 dense_52/Tensordot/axes:output:0'dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/concat°
dense_52/Tensordot/stackPack dense_52/Tensordot/Prod:output:0"dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/stackà
dense_52/Tensordot/transpose	Transposedense_51/Relu:activations:0"dense_52/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_52/Tensordot/transposeÃ
dense_52/Tensordot/ReshapeReshape dense_52/Tensordot/transpose:y:0!dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_52/Tensordot/ReshapeÂ
dense_52/Tensordot/MatMulMatMul#dense_52/Tensordot/Reshape:output:0)dense_52/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_52/Tensordot/MatMul
dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_52/Tensordot/Const_2
 dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_52/Tensordot/concat_1/axisê
dense_52/Tensordot/concat_1ConcatV2$dense_52/Tensordot/GatherV2:output:0#dense_52/Tensordot/Const_2:output:0)dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/concat_1Ô
dense_52/TensordotReshape#dense_52/Tensordot/MatMul:product:0$dense_52/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_52/Tensordot§
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_52/BiasAdd/ReadVariableOpË
dense_52/BiasAddBiasAdddense_52/Tensordot:output:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_52/BiasAdd
IdentityIdentitydense_52/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ:::::::s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿk

G__inference_sequential_22_layer_call_and_return_conditional_losses_7923

inputs.
*dense_50_tensordot_readvariableop_resource,
(dense_50_biasadd_readvariableop_resource.
*dense_51_tensordot_readvariableop_resource,
(dense_51_biasadd_readvariableop_resource.
*dense_52_tensordot_readvariableop_resource,
(dense_52_biasadd_readvariableop_resource
identity±
!dense_50/Tensordot/ReadVariableOpReadVariableOp*dense_50_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_50/Tensordot/ReadVariableOp|
dense_50/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_50/Tensordot/axes£
dense_50/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_50/Tensordot/freej
dense_50/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_50/Tensordot/Shape
 dense_50/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/GatherV2/axisþ
dense_50/Tensordot/GatherV2GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/free:output:0)dense_50/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_50/Tensordot/GatherV2
"dense_50/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_50/Tensordot/GatherV2_1/axis
dense_50/Tensordot/GatherV2_1GatherV2!dense_50/Tensordot/Shape:output:0 dense_50/Tensordot/axes:output:0+dense_50/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_50/Tensordot/GatherV2_1~
dense_50/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const¤
dense_50/Tensordot/ProdProd$dense_50/Tensordot/GatherV2:output:0!dense_50/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod
dense_50/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const_1¬
dense_50/Tensordot/Prod_1Prod&dense_50/Tensordot/GatherV2_1:output:0#dense_50/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_50/Tensordot/Prod_1
dense_50/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_50/Tensordot/concat/axisÝ
dense_50/Tensordot/concatConcatV2 dense_50/Tensordot/free:output:0 dense_50/Tensordot/axes:output:0'dense_50/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat°
dense_50/Tensordot/stackPack dense_50/Tensordot/Prod:output:0"dense_50/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/stackË
dense_50/Tensordot/transpose	Transposeinputs"dense_50/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_50/Tensordot/transposeÃ
dense_50/Tensordot/ReshapeReshape dense_50/Tensordot/transpose:y:0!dense_50/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_50/Tensordot/ReshapeÂ
dense_50/Tensordot/MatMulMatMul#dense_50/Tensordot/Reshape:output:0)dense_50/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_50/Tensordot/MatMul
dense_50/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_50/Tensordot/Const_2
 dense_50/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_50/Tensordot/concat_1/axisê
dense_50/Tensordot/concat_1ConcatV2$dense_50/Tensordot/GatherV2:output:0#dense_50/Tensordot/Const_2:output:0)dense_50/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_50/Tensordot/concat_1Ô
dense_50/TensordotReshape#dense_50/Tensordot/MatMul:product:0$dense_50/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_50/Tensordot§
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_50/BiasAdd/ReadVariableOpË
dense_50/BiasAddBiasAdddense_50/Tensordot:output:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_50/BiasAdd
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_50/Relu±
!dense_51/Tensordot/ReadVariableOpReadVariableOp*dense_51_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_51/Tensordot/ReadVariableOp|
dense_51/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_51/Tensordot/axes£
dense_51/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_51/Tensordot/free
dense_51/Tensordot/ShapeShapedense_50/Relu:activations:0*
T0*
_output_shapes
:2
dense_51/Tensordot/Shape
 dense_51/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_51/Tensordot/GatherV2/axisþ
dense_51/Tensordot/GatherV2GatherV2!dense_51/Tensordot/Shape:output:0 dense_51/Tensordot/free:output:0)dense_51/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_51/Tensordot/GatherV2
"dense_51/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_51/Tensordot/GatherV2_1/axis
dense_51/Tensordot/GatherV2_1GatherV2!dense_51/Tensordot/Shape:output:0 dense_51/Tensordot/axes:output:0+dense_51/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_51/Tensordot/GatherV2_1~
dense_51/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const¤
dense_51/Tensordot/ProdProd$dense_51/Tensordot/GatherV2:output:0!dense_51/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_51/Tensordot/Prod
dense_51/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const_1¬
dense_51/Tensordot/Prod_1Prod&dense_51/Tensordot/GatherV2_1:output:0#dense_51/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_51/Tensordot/Prod_1
dense_51/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_51/Tensordot/concat/axisÝ
dense_51/Tensordot/concatConcatV2 dense_51/Tensordot/free:output:0 dense_51/Tensordot/axes:output:0'dense_51/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/concat°
dense_51/Tensordot/stackPack dense_51/Tensordot/Prod:output:0"dense_51/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/stackà
dense_51/Tensordot/transpose	Transposedense_50/Relu:activations:0"dense_51/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_51/Tensordot/transposeÃ
dense_51/Tensordot/ReshapeReshape dense_51/Tensordot/transpose:y:0!dense_51/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_51/Tensordot/ReshapeÂ
dense_51/Tensordot/MatMulMatMul#dense_51/Tensordot/Reshape:output:0)dense_51/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_51/Tensordot/MatMul
dense_51/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_51/Tensordot/Const_2
 dense_51/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_51/Tensordot/concat_1/axisê
dense_51/Tensordot/concat_1ConcatV2$dense_51/Tensordot/GatherV2:output:0#dense_51/Tensordot/Const_2:output:0)dense_51/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_51/Tensordot/concat_1Ô
dense_51/TensordotReshape#dense_51/Tensordot/MatMul:product:0$dense_51/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_51/Tensordot§
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_51/BiasAdd/ReadVariableOpË
dense_51/BiasAddBiasAdddense_51/Tensordot:output:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_51/BiasAdd
dense_51/ReluReludense_51/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_51/Relu±
!dense_52/Tensordot/ReadVariableOpReadVariableOp*dense_52_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_52/Tensordot/ReadVariableOp|
dense_52/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_52/Tensordot/axes£
dense_52/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_52/Tensordot/free
dense_52/Tensordot/ShapeShapedense_51/Relu:activations:0*
T0*
_output_shapes
:2
dense_52/Tensordot/Shape
 dense_52/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_52/Tensordot/GatherV2/axisþ
dense_52/Tensordot/GatherV2GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/free:output:0)dense_52/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_52/Tensordot/GatherV2
"dense_52/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_52/Tensordot/GatherV2_1/axis
dense_52/Tensordot/GatherV2_1GatherV2!dense_52/Tensordot/Shape:output:0 dense_52/Tensordot/axes:output:0+dense_52/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_52/Tensordot/GatherV2_1~
dense_52/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_52/Tensordot/Const¤
dense_52/Tensordot/ProdProd$dense_52/Tensordot/GatherV2:output:0!dense_52/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_52/Tensordot/Prod
dense_52/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_52/Tensordot/Const_1¬
dense_52/Tensordot/Prod_1Prod&dense_52/Tensordot/GatherV2_1:output:0#dense_52/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_52/Tensordot/Prod_1
dense_52/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_52/Tensordot/concat/axisÝ
dense_52/Tensordot/concatConcatV2 dense_52/Tensordot/free:output:0 dense_52/Tensordot/axes:output:0'dense_52/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/concat°
dense_52/Tensordot/stackPack dense_52/Tensordot/Prod:output:0"dense_52/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/stackà
dense_52/Tensordot/transpose	Transposedense_51/Relu:activations:0"dense_52/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_52/Tensordot/transposeÃ
dense_52/Tensordot/ReshapeReshape dense_52/Tensordot/transpose:y:0!dense_52/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_52/Tensordot/ReshapeÂ
dense_52/Tensordot/MatMulMatMul#dense_52/Tensordot/Reshape:output:0)dense_52/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_52/Tensordot/MatMul
dense_52/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_52/Tensordot/Const_2
 dense_52/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_52/Tensordot/concat_1/axisê
dense_52/Tensordot/concat_1ConcatV2$dense_52/Tensordot/GatherV2:output:0#dense_52/Tensordot/Const_2:output:0)dense_52/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_52/Tensordot/concat_1Ô
dense_52/TensordotReshape#dense_52/Tensordot/MatMul:product:0$dense_52/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_52/Tensordot§
dense_52/BiasAdd/ReadVariableOpReadVariableOp(dense_52_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_52/BiasAdd/ReadVariableOpË
dense_52/BiasAddBiasAdddense_52/Tensordot:output:0'dense_52/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_52/BiasAdd
IdentityIdentitydense_52/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ:::::::s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ã
G__inference_sequential_22_layer_call_and_return_conditional_losses_7663
input_19
dense_50_7647
dense_50_7649
dense_51_7652
dense_51_7654
dense_52_7657
dense_52_7659
identity¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall´
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_50_7647dense_50_7649*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_50_layer_call_and_return_conditional_losses_75342"
 dense_50/StatefulPartitionedCallÕ
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_7652dense_51_7654*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_51_layer_call_and_return_conditional_losses_75812"
 dense_51/StatefulPartitionedCallÕ
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_7657dense_52_7659*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_52_layer_call_and_return_conditional_losses_76272"
 dense_52/StatefulPartitionedCall
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_19
à 
­
B__inference_dense_50_layer_call_and_return_conditional_losses_7988

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
Tensordot/axes
Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack°
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1°
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd|
ReluReluBiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ:::s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
|
'__inference_dense_52_layer_call_fn_8076

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_52_layer_call_and_return_conditional_losses_76272
StatefulPartitionedCall²
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

Á
G__inference_sequential_22_layer_call_and_return_conditional_losses_7721

inputs
dense_50_7705
dense_50_7707
dense_51_7710
dense_51_7712
dense_52_7715
dense_52_7717
identity¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall²
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50_7705dense_50_7707*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_50_layer_call_and_return_conditional_losses_75342"
 dense_50/StatefulPartitionedCallÕ
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_7710dense_51_7712*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_51_layer_call_and_return_conditional_losses_75812"
 dense_51/StatefulPartitionedCallÕ
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_7715dense_52_7717*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_52_layer_call_and_return_conditional_losses_76272"
 dense_52/StatefulPartitionedCall
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Á
G__inference_sequential_22_layer_call_and_return_conditional_losses_7685

inputs
dense_50_7669
dense_50_7671
dense_51_7674
dense_51_7676
dense_52_7679
dense_52_7681
identity¢ dense_50/StatefulPartitionedCall¢ dense_51/StatefulPartitionedCall¢ dense_52/StatefulPartitionedCall²
 dense_50/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50_7669dense_50_7671*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_50_layer_call_and_return_conditional_losses_75342"
 dense_50/StatefulPartitionedCallÕ
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_7674dense_51_7676*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_51_layer_call_and_return_conditional_losses_75812"
 dense_51/StatefulPartitionedCallÕ
 dense_52/StatefulPartitionedCallStatefulPartitionedCall)dense_51/StatefulPartitionedCall:output:0dense_52_7679dense_52_7681*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_52_layer_call_and_return_conditional_losses_76272"
 dense_52/StatefulPartitionedCall
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall!^dense_52/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
­
B__inference_dense_52_layer_call_and_return_conditional_losses_8067

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
Tensordot/axes
Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack°
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1°
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ :::s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
è
|
'__inference_dense_51_layer_call_fn_8037

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_51_layer_call_and_return_conditional_losses_75812
StatefulPartitionedCall²
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó
½
,__inference_sequential_22_layer_call_fn_7957

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÏ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_22_layer_call_and_return_conditional_losses_77212
StatefulPartitionedCall²
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à 
­
B__inference_dense_51_layer_call_and_return_conditional_losses_8028

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
Tensordot/axes
Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack°
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1°
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd|
ReluReluBiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ :::s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
è
|
'__inference_dense_50_layer_call_fn_7997

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_50_layer_call_and_return_conditional_losses_75342
StatefulPartitionedCall²
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
µ
"__inference_signature_wrapper_7755
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_74992
StatefulPartitionedCall²
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_19
à 
­
B__inference_dense_50_layer_call_and_return_conditional_losses_7534

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
Tensordot/axes
Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack°
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
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
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1°
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2	
BiasAdd|
ReluReluBiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
Relu
IdentityIdentityRelu:activations:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ:::s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
­
B__inference_dense_52_layer_call_and_return_conditional_losses_7627

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

: *
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
Tensordot/axes
Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
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
Tensordot/Const
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
Tensordot/Const_1
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
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack°
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1°
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp§
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2	
BiasAdd
IdentityIdentityBiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:5ÿÿÿÿÿÿÿÿÿ :::s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*õ
serving_defaultá
a
input_19U
serving_default_input_19:05ÿÿÿÿÿÿÿÿÿ`
dense_52T
StatefulPartitionedCall:05ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:÷
¯!
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api
	
signatures
0_default_save_signature
1__call__
*2&call_and_return_all_conditional_losses"ï
_tf_keras_sequentialÐ{"class_name": "Sequential", "name": "sequential_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}}, {"class_name": "Dense", "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}



kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
3__call__
*4&call_and_return_all_conditional_losses"æ
_tf_keras_layerÌ{"class_name": "Dense", "name": "dense_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_50", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"class_name": "Dense", "name": "dense_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_51", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 32]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
7__call__
*8&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "Dense", "name": "dense_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 32]}}
"
	optimizer
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
Ê
regularization_losses
metrics
non_trainable_variables
	variables
layer_regularization_losses
layer_metrics
trainable_variables

 layers
1__call__
0_default_save_signature
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
,
9serving_default"
signature_map
!: 2dense_50/kernel
: 2dense_50/bias
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
­
regularization_losses
!metrics
"non_trainable_variables
#layer_regularization_losses
	variables
$layer_metrics
trainable_variables

%layers
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
!:  2dense_51/kernel
: 2dense_51/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
&metrics
'non_trainable_variables
(layer_regularization_losses
	variables
)layer_metrics
trainable_variables

*layers
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_52/kernel
:2dense_52/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
+metrics
,non_trainable_variables
-layer_regularization_losses
	variables
.layer_metrics
trainable_variables

/layers
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
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
2ÿ
__inference__wrapped_model_7499Û
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *K¢H
FC
input_195ÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_sequential_22_layer_call_fn_7736
,__inference_sequential_22_layer_call_fn_7940
,__inference_sequential_22_layer_call_fn_7957
,__inference_sequential_22_layer_call_fn_7700À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_22_layer_call_and_return_conditional_losses_7839
G__inference_sequential_22_layer_call_and_return_conditional_losses_7663
G__inference_sequential_22_layer_call_and_return_conditional_losses_7923
G__inference_sequential_22_layer_call_and_return_conditional_losses_7644À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_dense_50_layer_call_fn_7997¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_50_layer_call_and_return_conditional_losses_7988¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_51_layer_call_fn_8037¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_51_layer_call_and_return_conditional_losses_8028¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_52_layer_call_fn_8076¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_52_layer_call_and_return_conditional_losses_8067¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2B0
"__inference_signature_wrapper_7755input_19Ü
__inference__wrapped_model_7499¸
U¢R
K¢H
FC
input_195ÿÿÿÿÿÿÿÿÿ
ª "WªT
R
dense_52FC
dense_525ÿÿÿÿÿÿÿÿÿë
B__inference_dense_50_layer_call_and_return_conditional_losses_7988¤
S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ 
 Ã
'__inference_dense_50_layer_call_fn_7997
S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ
ª "<95ÿÿÿÿÿÿÿÿÿ ë
B__inference_dense_51_layer_call_and_return_conditional_losses_8028¤S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ 
 Ã
'__inference_dense_51_layer_call_fn_8037S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "<95ÿÿÿÿÿÿÿÿÿ ë
B__inference_dense_52_layer_call_and_return_conditional_losses_8067¤S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 Ã
'__inference_dense_52_layer_call_fn_8076S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "<95ÿÿÿÿÿÿÿÿÿþ
G__inference_sequential_22_layer_call_and_return_conditional_losses_7644²
]¢Z
S¢P
FC
input_195ÿÿÿÿÿÿÿÿÿ
p

 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 þ
G__inference_sequential_22_layer_call_and_return_conditional_losses_7663²
]¢Z
S¢P
FC
input_195ÿÿÿÿÿÿÿÿÿ
p 

 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 ü
G__inference_sequential_22_layer_call_and_return_conditional_losses_7839°
[¢X
Q¢N
DA
inputs5ÿÿÿÿÿÿÿÿÿ
p

 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 ü
G__inference_sequential_22_layer_call_and_return_conditional_losses_7923°
[¢X
Q¢N
DA
inputs5ÿÿÿÿÿÿÿÿÿ
p 

 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 Ö
,__inference_sequential_22_layer_call_fn_7700¥
]¢Z
S¢P
FC
input_195ÿÿÿÿÿÿÿÿÿ
p

 
ª "<95ÿÿÿÿÿÿÿÿÿÖ
,__inference_sequential_22_layer_call_fn_7736¥
]¢Z
S¢P
FC
input_195ÿÿÿÿÿÿÿÿÿ
p 

 
ª "<95ÿÿÿÿÿÿÿÿÿÔ
,__inference_sequential_22_layer_call_fn_7940£
[¢X
Q¢N
DA
inputs5ÿÿÿÿÿÿÿÿÿ
p

 
ª "<95ÿÿÿÿÿÿÿÿÿÔ
,__inference_sequential_22_layer_call_fn_7957£
[¢X
Q¢N
DA
inputs5ÿÿÿÿÿÿÿÿÿ
p 

 
ª "<95ÿÿÿÿÿÿÿÿÿë
"__inference_signature_wrapper_7755Ä
a¢^
¢ 
WªT
R
input_19FC
input_195ÿÿÿÿÿÿÿÿÿ"WªT
R
dense_52FC
dense_525ÿÿÿÿÿÿÿÿÿ