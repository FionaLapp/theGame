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
dense_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_38/kernel
s
#dense_38/kernel/Read/ReadVariableOpReadVariableOpdense_38/kernel*
_output_shapes

: *
dtype0
r
dense_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_38/bias
k
!dense_38/bias/Read/ReadVariableOpReadVariableOpdense_38/bias*
_output_shapes
: *
dtype0
z
dense_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_39/kernel
s
#dense_39/kernel/Read/ReadVariableOpReadVariableOpdense_39/kernel*
_output_shapes

:  *
dtype0
r
dense_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_39/bias
k
!dense_39/bias/Read/ReadVariableOpReadVariableOpdense_39/bias*
_output_shapes
: *
dtype0
z
dense_40/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_40/kernel
s
#dense_40/kernel/Read/ReadVariableOpReadVariableOpdense_40/kernel*
_output_shapes

: *
dtype0
r
dense_40/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_40/bias
k
!dense_40/bias/Read/ReadVariableOpReadVariableOpdense_40/bias*
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
VARIABLE_VALUEdense_38/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_38/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_39/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_39/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_40/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_40/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_input_15Placeholder*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*
dtype0*@
shape7:5ÿÿÿÿÿÿÿÿÿ
Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_15dense_38/kerneldense_38/biasdense_39/kerneldense_39/biasdense_40/kerneldense_40/bias*
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
"__inference_signature_wrapper_5452
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_38/kernel/Read/ReadVariableOp!dense_38/bias/Read/ReadVariableOp#dense_39/kernel/Read/ReadVariableOp!dense_39/bias/Read/ReadVariableOp#dense_40/kernel/Read/ReadVariableOp!dense_40/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_5814
ú
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_38/kerneldense_38/biasdense_39/kerneldense_39/biasdense_40/kerneldense_40/bias*
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
 __inference__traced_restore_5842ëë
è
|
'__inference_dense_39_layer_call_fn_5734

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
B__inference_dense_39_layer_call_and_return_conditional_losses_52782
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
ù

__inference__traced_save_5814
file_prefix.
*savev2_dense_38_kernel_read_readvariableop,
(savev2_dense_38_bias_read_readvariableop.
*savev2_dense_39_kernel_read_readvariableop,
(savev2_dense_39_bias_read_readvariableop.
*savev2_dense_40_kernel_read_readvariableop,
(savev2_dense_40_bias_read_readvariableop
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
value3B1 B+_temp_9987e6f687b54329aa4d89e47c131852/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_38_kernel_read_readvariableop(savev2_dense_38_bias_read_readvariableop*savev2_dense_39_kernel_read_readvariableop(savev2_dense_39_bias_read_readvariableop*savev2_dense_40_kernel_read_readvariableop(savev2_dense_40_bias_read_readvariableopsavev2_const"/device:CPU:0*
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

Á
G__inference_sequential_18_layer_call_and_return_conditional_losses_5382

inputs
dense_38_5366
dense_38_5368
dense_39_5371
dense_39_5373
dense_40_5376
dense_40_5378
identity¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall²
 dense_38/StatefulPartitionedCallStatefulPartitionedCallinputsdense_38_5366dense_38_5368*
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
B__inference_dense_38_layer_call_and_return_conditional_losses_52312"
 dense_38/StatefulPartitionedCallÕ
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_5371dense_39_5373*
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
B__inference_dense_39_layer_call_and_return_conditional_losses_52782"
 dense_39/StatefulPartitionedCallÕ
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_5376dense_40_5378*
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
B__inference_dense_40_layer_call_and_return_conditional_losses_53242"
 dense_40/StatefulPartitionedCall
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
­
B__inference_dense_40_layer_call_and_return_conditional_losses_5764

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

Á
G__inference_sequential_18_layer_call_and_return_conditional_losses_5418

inputs
dense_38_5402
dense_38_5404
dense_39_5407
dense_39_5409
dense_40_5412
dense_40_5414
identity¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall²
 dense_38/StatefulPartitionedCallStatefulPartitionedCallinputsdense_38_5402dense_38_5404*
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
B__inference_dense_38_layer_call_and_return_conditional_losses_52312"
 dense_38/StatefulPartitionedCallÕ
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_5407dense_39_5409*
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
B__inference_dense_39_layer_call_and_return_conditional_losses_52782"
 dense_39/StatefulPartitionedCallÕ
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_5412dense_40_5414*
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
B__inference_dense_40_layer_call_and_return_conditional_losses_53242"
 dense_40/StatefulPartitionedCall
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
µ
"__inference_signature_wrapper_5452
input_15
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
__inference__wrapped_model_51962
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
input_15
ÿk

G__inference_sequential_18_layer_call_and_return_conditional_losses_5620

inputs.
*dense_38_tensordot_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource.
*dense_39_tensordot_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource.
*dense_40_tensordot_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource
identity±
!dense_38/Tensordot/ReadVariableOpReadVariableOp*dense_38_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_38/Tensordot/ReadVariableOp|
dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_38/Tensordot/axes£
dense_38/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_38/Tensordot/freej
dense_38/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_38/Tensordot/Shape
 dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/GatherV2/axisþ
dense_38/Tensordot/GatherV2GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/free:output:0)dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_38/Tensordot/GatherV2
"dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_38/Tensordot/GatherV2_1/axis
dense_38/Tensordot/GatherV2_1GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/axes:output:0+dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2_1~
dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const¤
dense_38/Tensordot/ProdProd$dense_38/Tensordot/GatherV2:output:0!dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod
dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_1¬
dense_38/Tensordot/Prod_1Prod&dense_38/Tensordot/GatherV2_1:output:0#dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod_1
dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_38/Tensordot/concat/axisÝ
dense_38/Tensordot/concatConcatV2 dense_38/Tensordot/free:output:0 dense_38/Tensordot/axes:output:0'dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat°
dense_38/Tensordot/stackPack dense_38/Tensordot/Prod:output:0"dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/stackË
dense_38/Tensordot/transpose	Transposeinputs"dense_38/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_38/Tensordot/transposeÃ
dense_38/Tensordot/ReshapeReshape dense_38/Tensordot/transpose:y:0!dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_38/Tensordot/ReshapeÂ
dense_38/Tensordot/MatMulMatMul#dense_38/Tensordot/Reshape:output:0)dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/Tensordot/MatMul
dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_2
 dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/concat_1/axisê
dense_38/Tensordot/concat_1ConcatV2$dense_38/Tensordot/GatherV2:output:0#dense_38/Tensordot/Const_2:output:0)dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat_1Ô
dense_38/TensordotReshape#dense_38/Tensordot/MatMul:product:0$dense_38/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_38/Tensordot§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOpË
dense_38/BiasAddBiasAdddense_38/Tensordot:output:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_38/BiasAdd
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_38/Relu±
!dense_39/Tensordot/ReadVariableOpReadVariableOp*dense_39_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_39/Tensordot/ReadVariableOp|
dense_39/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_39/Tensordot/axes£
dense_39/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_39/Tensordot/free
dense_39/Tensordot/ShapeShapedense_38/Relu:activations:0*
T0*
_output_shapes
:2
dense_39/Tensordot/Shape
 dense_39/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_39/Tensordot/GatherV2/axisþ
dense_39/Tensordot/GatherV2GatherV2!dense_39/Tensordot/Shape:output:0 dense_39/Tensordot/free:output:0)dense_39/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_39/Tensordot/GatherV2
"dense_39/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_39/Tensordot/GatherV2_1/axis
dense_39/Tensordot/GatherV2_1GatherV2!dense_39/Tensordot/Shape:output:0 dense_39/Tensordot/axes:output:0+dense_39/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_39/Tensordot/GatherV2_1~
dense_39/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_39/Tensordot/Const¤
dense_39/Tensordot/ProdProd$dense_39/Tensordot/GatherV2:output:0!dense_39/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_39/Tensordot/Prod
dense_39/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_39/Tensordot/Const_1¬
dense_39/Tensordot/Prod_1Prod&dense_39/Tensordot/GatherV2_1:output:0#dense_39/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_39/Tensordot/Prod_1
dense_39/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_39/Tensordot/concat/axisÝ
dense_39/Tensordot/concatConcatV2 dense_39/Tensordot/free:output:0 dense_39/Tensordot/axes:output:0'dense_39/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_39/Tensordot/concat°
dense_39/Tensordot/stackPack dense_39/Tensordot/Prod:output:0"dense_39/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_39/Tensordot/stackà
dense_39/Tensordot/transpose	Transposedense_38/Relu:activations:0"dense_39/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_39/Tensordot/transposeÃ
dense_39/Tensordot/ReshapeReshape dense_39/Tensordot/transpose:y:0!dense_39/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_39/Tensordot/ReshapeÂ
dense_39/Tensordot/MatMulMatMul#dense_39/Tensordot/Reshape:output:0)dense_39/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_39/Tensordot/MatMul
dense_39/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_39/Tensordot/Const_2
 dense_39/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_39/Tensordot/concat_1/axisê
dense_39/Tensordot/concat_1ConcatV2$dense_39/Tensordot/GatherV2:output:0#dense_39/Tensordot/Const_2:output:0)dense_39/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_39/Tensordot/concat_1Ô
dense_39/TensordotReshape#dense_39/Tensordot/MatMul:product:0$dense_39/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_39/Tensordot§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_39/BiasAdd/ReadVariableOpË
dense_39/BiasAddBiasAdddense_39/Tensordot:output:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_39/BiasAdd
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_39/Relu±
!dense_40/Tensordot/ReadVariableOpReadVariableOp*dense_40_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_40/Tensordot/ReadVariableOp|
dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_40/Tensordot/axes£
dense_40/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_40/Tensordot/free
dense_40/Tensordot/ShapeShapedense_39/Relu:activations:0*
T0*
_output_shapes
:2
dense_40/Tensordot/Shape
 dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_40/Tensordot/GatherV2/axisþ
dense_40/Tensordot/GatherV2GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/free:output:0)dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_40/Tensordot/GatherV2
"dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_40/Tensordot/GatherV2_1/axis
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
dense_40/Tensordot/Const¤
dense_40/Tensordot/ProdProd$dense_40/Tensordot/GatherV2:output:0!dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_40/Tensordot/Prod
dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_40/Tensordot/Const_1¬
dense_40/Tensordot/Prod_1Prod&dense_40/Tensordot/GatherV2_1:output:0#dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_40/Tensordot/Prod_1
dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_40/Tensordot/concat/axisÝ
dense_40/Tensordot/concatConcatV2 dense_40/Tensordot/free:output:0 dense_40/Tensordot/axes:output:0'dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/concat°
dense_40/Tensordot/stackPack dense_40/Tensordot/Prod:output:0"dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/stackà
dense_40/Tensordot/transpose	Transposedense_39/Relu:activations:0"dense_40/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_40/Tensordot/transposeÃ
dense_40/Tensordot/ReshapeReshape dense_40/Tensordot/transpose:y:0!dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_40/Tensordot/ReshapeÂ
dense_40/Tensordot/MatMulMatMul#dense_40/Tensordot/Reshape:output:0)dense_40/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_40/Tensordot/MatMul
dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_40/Tensordot/Const_2
 dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_40/Tensordot/concat_1/axisê
dense_40/Tensordot/concat_1ConcatV2$dense_40/Tensordot/GatherV2:output:0#dense_40/Tensordot/Const_2:output:0)dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/concat_1Ô
dense_40/TensordotReshape#dense_40/Tensordot/MatMul:product:0$dense_40/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_40/Tensordot§
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_40/BiasAdd/ReadVariableOpË
dense_40/BiasAddBiasAdddense_40/Tensordot:output:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_40/BiasAdd
IdentityIdentitydense_40/BiasAdd:output:0*
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
ù
¿
,__inference_sequential_18_layer_call_fn_5433
input_15
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
G__inference_sequential_18_layer_call_and_return_conditional_losses_54182
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
input_15
ù
¿
,__inference_sequential_18_layer_call_fn_5397
input_15
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
G__inference_sequential_18_layer_call_and_return_conditional_losses_53822
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
input_15
à 
­
B__inference_dense_38_layer_call_and_return_conditional_losses_5231

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
ó
½
,__inference_sequential_18_layer_call_fn_5654

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
G__inference_sequential_18_layer_call_and_return_conditional_losses_54182
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
ó
½
,__inference_sequential_18_layer_call_fn_5637

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
G__inference_sequential_18_layer_call_and_return_conditional_losses_53822
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
è
|
'__inference_dense_38_layer_call_fn_5694

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
B__inference_dense_38_layer_call_and_return_conditional_losses_52312
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
à
­
B__inference_dense_40_layer_call_and_return_conditional_losses_5324

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
ÿk

G__inference_sequential_18_layer_call_and_return_conditional_losses_5536

inputs.
*dense_38_tensordot_readvariableop_resource,
(dense_38_biasadd_readvariableop_resource.
*dense_39_tensordot_readvariableop_resource,
(dense_39_biasadd_readvariableop_resource.
*dense_40_tensordot_readvariableop_resource,
(dense_40_biasadd_readvariableop_resource
identity±
!dense_38/Tensordot/ReadVariableOpReadVariableOp*dense_38_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_38/Tensordot/ReadVariableOp|
dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_38/Tensordot/axes£
dense_38/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_38/Tensordot/freej
dense_38/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_38/Tensordot/Shape
 dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/GatherV2/axisþ
dense_38/Tensordot/GatherV2GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/free:output:0)dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_38/Tensordot/GatherV2
"dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_38/Tensordot/GatherV2_1/axis
dense_38/Tensordot/GatherV2_1GatherV2!dense_38/Tensordot/Shape:output:0 dense_38/Tensordot/axes:output:0+dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_38/Tensordot/GatherV2_1~
dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const¤
dense_38/Tensordot/ProdProd$dense_38/Tensordot/GatherV2:output:0!dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod
dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_1¬
dense_38/Tensordot/Prod_1Prod&dense_38/Tensordot/GatherV2_1:output:0#dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_38/Tensordot/Prod_1
dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_38/Tensordot/concat/axisÝ
dense_38/Tensordot/concatConcatV2 dense_38/Tensordot/free:output:0 dense_38/Tensordot/axes:output:0'dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat°
dense_38/Tensordot/stackPack dense_38/Tensordot/Prod:output:0"dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/stackË
dense_38/Tensordot/transpose	Transposeinputs"dense_38/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_38/Tensordot/transposeÃ
dense_38/Tensordot/ReshapeReshape dense_38/Tensordot/transpose:y:0!dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_38/Tensordot/ReshapeÂ
dense_38/Tensordot/MatMulMatMul#dense_38/Tensordot/Reshape:output:0)dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_38/Tensordot/MatMul
dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_38/Tensordot/Const_2
 dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_38/Tensordot/concat_1/axisê
dense_38/Tensordot/concat_1ConcatV2$dense_38/Tensordot/GatherV2:output:0#dense_38/Tensordot/Const_2:output:0)dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_38/Tensordot/concat_1Ô
dense_38/TensordotReshape#dense_38/Tensordot/MatMul:product:0$dense_38/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_38/Tensordot§
dense_38/BiasAdd/ReadVariableOpReadVariableOp(dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_38/BiasAdd/ReadVariableOpË
dense_38/BiasAddBiasAdddense_38/Tensordot:output:0'dense_38/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_38/BiasAdd
dense_38/ReluReludense_38/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_38/Relu±
!dense_39/Tensordot/ReadVariableOpReadVariableOp*dense_39_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_39/Tensordot/ReadVariableOp|
dense_39/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_39/Tensordot/axes£
dense_39/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_39/Tensordot/free
dense_39/Tensordot/ShapeShapedense_38/Relu:activations:0*
T0*
_output_shapes
:2
dense_39/Tensordot/Shape
 dense_39/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_39/Tensordot/GatherV2/axisþ
dense_39/Tensordot/GatherV2GatherV2!dense_39/Tensordot/Shape:output:0 dense_39/Tensordot/free:output:0)dense_39/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_39/Tensordot/GatherV2
"dense_39/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_39/Tensordot/GatherV2_1/axis
dense_39/Tensordot/GatherV2_1GatherV2!dense_39/Tensordot/Shape:output:0 dense_39/Tensordot/axes:output:0+dense_39/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_39/Tensordot/GatherV2_1~
dense_39/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_39/Tensordot/Const¤
dense_39/Tensordot/ProdProd$dense_39/Tensordot/GatherV2:output:0!dense_39/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_39/Tensordot/Prod
dense_39/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_39/Tensordot/Const_1¬
dense_39/Tensordot/Prod_1Prod&dense_39/Tensordot/GatherV2_1:output:0#dense_39/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_39/Tensordot/Prod_1
dense_39/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_39/Tensordot/concat/axisÝ
dense_39/Tensordot/concatConcatV2 dense_39/Tensordot/free:output:0 dense_39/Tensordot/axes:output:0'dense_39/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_39/Tensordot/concat°
dense_39/Tensordot/stackPack dense_39/Tensordot/Prod:output:0"dense_39/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_39/Tensordot/stackà
dense_39/Tensordot/transpose	Transposedense_38/Relu:activations:0"dense_39/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_39/Tensordot/transposeÃ
dense_39/Tensordot/ReshapeReshape dense_39/Tensordot/transpose:y:0!dense_39/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_39/Tensordot/ReshapeÂ
dense_39/Tensordot/MatMulMatMul#dense_39/Tensordot/Reshape:output:0)dense_39/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_39/Tensordot/MatMul
dense_39/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_39/Tensordot/Const_2
 dense_39/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_39/Tensordot/concat_1/axisê
dense_39/Tensordot/concat_1ConcatV2$dense_39/Tensordot/GatherV2:output:0#dense_39/Tensordot/Const_2:output:0)dense_39/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_39/Tensordot/concat_1Ô
dense_39/TensordotReshape#dense_39/Tensordot/MatMul:product:0$dense_39/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_39/Tensordot§
dense_39/BiasAdd/ReadVariableOpReadVariableOp(dense_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_39/BiasAdd/ReadVariableOpË
dense_39/BiasAddBiasAdddense_39/Tensordot:output:0'dense_39/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_39/BiasAdd
dense_39/ReluReludense_39/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_39/Relu±
!dense_40/Tensordot/ReadVariableOpReadVariableOp*dense_40_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_40/Tensordot/ReadVariableOp|
dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_40/Tensordot/axes£
dense_40/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_40/Tensordot/free
dense_40/Tensordot/ShapeShapedense_39/Relu:activations:0*
T0*
_output_shapes
:2
dense_40/Tensordot/Shape
 dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_40/Tensordot/GatherV2/axisþ
dense_40/Tensordot/GatherV2GatherV2!dense_40/Tensordot/Shape:output:0 dense_40/Tensordot/free:output:0)dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_40/Tensordot/GatherV2
"dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_40/Tensordot/GatherV2_1/axis
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
dense_40/Tensordot/Const¤
dense_40/Tensordot/ProdProd$dense_40/Tensordot/GatherV2:output:0!dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_40/Tensordot/Prod
dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_40/Tensordot/Const_1¬
dense_40/Tensordot/Prod_1Prod&dense_40/Tensordot/GatherV2_1:output:0#dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_40/Tensordot/Prod_1
dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_40/Tensordot/concat/axisÝ
dense_40/Tensordot/concatConcatV2 dense_40/Tensordot/free:output:0 dense_40/Tensordot/axes:output:0'dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/concat°
dense_40/Tensordot/stackPack dense_40/Tensordot/Prod:output:0"dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/stackà
dense_40/Tensordot/transpose	Transposedense_39/Relu:activations:0"dense_40/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_40/Tensordot/transposeÃ
dense_40/Tensordot/ReshapeReshape dense_40/Tensordot/transpose:y:0!dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_40/Tensordot/ReshapeÂ
dense_40/Tensordot/MatMulMatMul#dense_40/Tensordot/Reshape:output:0)dense_40/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_40/Tensordot/MatMul
dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_40/Tensordot/Const_2
 dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_40/Tensordot/concat_1/axisê
dense_40/Tensordot/concat_1ConcatV2$dense_40/Tensordot/GatherV2:output:0#dense_40/Tensordot/Const_2:output:0)dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_40/Tensordot/concat_1Ô
dense_40/TensordotReshape#dense_40/Tensordot/MatMul:product:0$dense_40/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_40/Tensordot§
dense_40/BiasAdd/ReadVariableOpReadVariableOp(dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_40/BiasAdd/ReadVariableOpË
dense_40/BiasAddBiasAdddense_40/Tensordot:output:0'dense_40/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_40/BiasAdd
IdentityIdentitydense_40/BiasAdd:output:0*
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
à 
­
B__inference_dense_39_layer_call_and_return_conditional_losses_5278

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
à 
­
B__inference_dense_38_layer_call_and_return_conditional_losses_5685

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
ø
®
__inference__wrapped_model_5196
input_15<
8sequential_18_dense_38_tensordot_readvariableop_resource:
6sequential_18_dense_38_biasadd_readvariableop_resource<
8sequential_18_dense_39_tensordot_readvariableop_resource:
6sequential_18_dense_39_biasadd_readvariableop_resource<
8sequential_18_dense_40_tensordot_readvariableop_resource:
6sequential_18_dense_40_biasadd_readvariableop_resource
identityÛ
/sequential_18/dense_38/Tensordot/ReadVariableOpReadVariableOp8sequential_18_dense_38_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_18/dense_38/Tensordot/ReadVariableOp
%sequential_18/dense_38/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_18/dense_38/Tensordot/axes¿
%sequential_18/dense_38/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_18/dense_38/Tensordot/free
&sequential_18/dense_38/Tensordot/ShapeShapeinput_15*
T0*
_output_shapes
:2(
&sequential_18/dense_38/Tensordot/Shape¢
.sequential_18/dense_38/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_38/Tensordot/GatherV2/axisÄ
)sequential_18/dense_38/Tensordot/GatherV2GatherV2/sequential_18/dense_38/Tensordot/Shape:output:0.sequential_18/dense_38/Tensordot/free:output:07sequential_18/dense_38/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_18/dense_38/Tensordot/GatherV2¦
0sequential_18/dense_38/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_18/dense_38/Tensordot/GatherV2_1/axisÊ
+sequential_18/dense_38/Tensordot/GatherV2_1GatherV2/sequential_18/dense_38/Tensordot/Shape:output:0.sequential_18/dense_38/Tensordot/axes:output:09sequential_18/dense_38/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_18/dense_38/Tensordot/GatherV2_1
&sequential_18/dense_38/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_18/dense_38/Tensordot/ConstÜ
%sequential_18/dense_38/Tensordot/ProdProd2sequential_18/dense_38/Tensordot/GatherV2:output:0/sequential_18/dense_38/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_18/dense_38/Tensordot/Prod
(sequential_18/dense_38/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_18/dense_38/Tensordot/Const_1ä
'sequential_18/dense_38/Tensordot/Prod_1Prod4sequential_18/dense_38/Tensordot/GatherV2_1:output:01sequential_18/dense_38/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_18/dense_38/Tensordot/Prod_1
,sequential_18/dense_38/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_18/dense_38/Tensordot/concat/axis£
'sequential_18/dense_38/Tensordot/concatConcatV2.sequential_18/dense_38/Tensordot/free:output:0.sequential_18/dense_38/Tensordot/axes:output:05sequential_18/dense_38/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_18/dense_38/Tensordot/concatè
&sequential_18/dense_38/Tensordot/stackPack.sequential_18/dense_38/Tensordot/Prod:output:00sequential_18/dense_38/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_18/dense_38/Tensordot/stack÷
*sequential_18/dense_38/Tensordot/transpose	Transposeinput_150sequential_18/dense_38/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2,
*sequential_18/dense_38/Tensordot/transposeû
(sequential_18/dense_38/Tensordot/ReshapeReshape.sequential_18/dense_38/Tensordot/transpose:y:0/sequential_18/dense_38/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_18/dense_38/Tensordot/Reshapeú
'sequential_18/dense_38/Tensordot/MatMulMatMul1sequential_18/dense_38/Tensordot/Reshape:output:07sequential_18/dense_38/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_18/dense_38/Tensordot/MatMul
(sequential_18/dense_38/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_18/dense_38/Tensordot/Const_2¢
.sequential_18/dense_38/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_38/Tensordot/concat_1/axis°
)sequential_18/dense_38/Tensordot/concat_1ConcatV22sequential_18/dense_38/Tensordot/GatherV2:output:01sequential_18/dense_38/Tensordot/Const_2:output:07sequential_18/dense_38/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_18/dense_38/Tensordot/concat_1
 sequential_18/dense_38/TensordotReshape1sequential_18/dense_38/Tensordot/MatMul:product:02sequential_18/dense_38/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2"
 sequential_18/dense_38/TensordotÑ
-sequential_18/dense_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_38_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_18/dense_38/BiasAdd/ReadVariableOp
sequential_18/dense_38/BiasAddBiasAdd)sequential_18/dense_38/Tensordot:output:05sequential_18/dense_38/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2 
sequential_18/dense_38/BiasAddÁ
sequential_18/dense_38/ReluRelu'sequential_18/dense_38/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
sequential_18/dense_38/ReluÛ
/sequential_18/dense_39/Tensordot/ReadVariableOpReadVariableOp8sequential_18_dense_39_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype021
/sequential_18/dense_39/Tensordot/ReadVariableOp
%sequential_18/dense_39/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_18/dense_39/Tensordot/axes¿
%sequential_18/dense_39/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_18/dense_39/Tensordot/free©
&sequential_18/dense_39/Tensordot/ShapeShape)sequential_18/dense_38/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_18/dense_39/Tensordot/Shape¢
.sequential_18/dense_39/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_39/Tensordot/GatherV2/axisÄ
)sequential_18/dense_39/Tensordot/GatherV2GatherV2/sequential_18/dense_39/Tensordot/Shape:output:0.sequential_18/dense_39/Tensordot/free:output:07sequential_18/dense_39/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_18/dense_39/Tensordot/GatherV2¦
0sequential_18/dense_39/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_18/dense_39/Tensordot/GatherV2_1/axisÊ
+sequential_18/dense_39/Tensordot/GatherV2_1GatherV2/sequential_18/dense_39/Tensordot/Shape:output:0.sequential_18/dense_39/Tensordot/axes:output:09sequential_18/dense_39/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_18/dense_39/Tensordot/GatherV2_1
&sequential_18/dense_39/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_18/dense_39/Tensordot/ConstÜ
%sequential_18/dense_39/Tensordot/ProdProd2sequential_18/dense_39/Tensordot/GatherV2:output:0/sequential_18/dense_39/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_18/dense_39/Tensordot/Prod
(sequential_18/dense_39/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_18/dense_39/Tensordot/Const_1ä
'sequential_18/dense_39/Tensordot/Prod_1Prod4sequential_18/dense_39/Tensordot/GatherV2_1:output:01sequential_18/dense_39/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_18/dense_39/Tensordot/Prod_1
,sequential_18/dense_39/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_18/dense_39/Tensordot/concat/axis£
'sequential_18/dense_39/Tensordot/concatConcatV2.sequential_18/dense_39/Tensordot/free:output:0.sequential_18/dense_39/Tensordot/axes:output:05sequential_18/dense_39/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_18/dense_39/Tensordot/concatè
&sequential_18/dense_39/Tensordot/stackPack.sequential_18/dense_39/Tensordot/Prod:output:00sequential_18/dense_39/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_18/dense_39/Tensordot/stack
*sequential_18/dense_39/Tensordot/transpose	Transpose)sequential_18/dense_38/Relu:activations:00sequential_18/dense_39/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2,
*sequential_18/dense_39/Tensordot/transposeû
(sequential_18/dense_39/Tensordot/ReshapeReshape.sequential_18/dense_39/Tensordot/transpose:y:0/sequential_18/dense_39/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_18/dense_39/Tensordot/Reshapeú
'sequential_18/dense_39/Tensordot/MatMulMatMul1sequential_18/dense_39/Tensordot/Reshape:output:07sequential_18/dense_39/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_18/dense_39/Tensordot/MatMul
(sequential_18/dense_39/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_18/dense_39/Tensordot/Const_2¢
.sequential_18/dense_39/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_39/Tensordot/concat_1/axis°
)sequential_18/dense_39/Tensordot/concat_1ConcatV22sequential_18/dense_39/Tensordot/GatherV2:output:01sequential_18/dense_39/Tensordot/Const_2:output:07sequential_18/dense_39/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_18/dense_39/Tensordot/concat_1
 sequential_18/dense_39/TensordotReshape1sequential_18/dense_39/Tensordot/MatMul:product:02sequential_18/dense_39/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2"
 sequential_18/dense_39/TensordotÑ
-sequential_18/dense_39/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_39_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_18/dense_39/BiasAdd/ReadVariableOp
sequential_18/dense_39/BiasAddBiasAdd)sequential_18/dense_39/Tensordot:output:05sequential_18/dense_39/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2 
sequential_18/dense_39/BiasAddÁ
sequential_18/dense_39/ReluRelu'sequential_18/dense_39/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
sequential_18/dense_39/ReluÛ
/sequential_18/dense_40/Tensordot/ReadVariableOpReadVariableOp8sequential_18_dense_40_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_18/dense_40/Tensordot/ReadVariableOp
%sequential_18/dense_40/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_18/dense_40/Tensordot/axes¿
%sequential_18/dense_40/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_18/dense_40/Tensordot/free©
&sequential_18/dense_40/Tensordot/ShapeShape)sequential_18/dense_39/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_18/dense_40/Tensordot/Shape¢
.sequential_18/dense_40/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_40/Tensordot/GatherV2/axisÄ
)sequential_18/dense_40/Tensordot/GatherV2GatherV2/sequential_18/dense_40/Tensordot/Shape:output:0.sequential_18/dense_40/Tensordot/free:output:07sequential_18/dense_40/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_18/dense_40/Tensordot/GatherV2¦
0sequential_18/dense_40/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_18/dense_40/Tensordot/GatherV2_1/axisÊ
+sequential_18/dense_40/Tensordot/GatherV2_1GatherV2/sequential_18/dense_40/Tensordot/Shape:output:0.sequential_18/dense_40/Tensordot/axes:output:09sequential_18/dense_40/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_18/dense_40/Tensordot/GatherV2_1
&sequential_18/dense_40/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_18/dense_40/Tensordot/ConstÜ
%sequential_18/dense_40/Tensordot/ProdProd2sequential_18/dense_40/Tensordot/GatherV2:output:0/sequential_18/dense_40/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_18/dense_40/Tensordot/Prod
(sequential_18/dense_40/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_18/dense_40/Tensordot/Const_1ä
'sequential_18/dense_40/Tensordot/Prod_1Prod4sequential_18/dense_40/Tensordot/GatherV2_1:output:01sequential_18/dense_40/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_18/dense_40/Tensordot/Prod_1
,sequential_18/dense_40/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_18/dense_40/Tensordot/concat/axis£
'sequential_18/dense_40/Tensordot/concatConcatV2.sequential_18/dense_40/Tensordot/free:output:0.sequential_18/dense_40/Tensordot/axes:output:05sequential_18/dense_40/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_18/dense_40/Tensordot/concatè
&sequential_18/dense_40/Tensordot/stackPack.sequential_18/dense_40/Tensordot/Prod:output:00sequential_18/dense_40/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_18/dense_40/Tensordot/stack
*sequential_18/dense_40/Tensordot/transpose	Transpose)sequential_18/dense_39/Relu:activations:00sequential_18/dense_40/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2,
*sequential_18/dense_40/Tensordot/transposeû
(sequential_18/dense_40/Tensordot/ReshapeReshape.sequential_18/dense_40/Tensordot/transpose:y:0/sequential_18/dense_40/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_18/dense_40/Tensordot/Reshapeú
'sequential_18/dense_40/Tensordot/MatMulMatMul1sequential_18/dense_40/Tensordot/Reshape:output:07sequential_18/dense_40/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_18/dense_40/Tensordot/MatMul
(sequential_18/dense_40/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_18/dense_40/Tensordot/Const_2¢
.sequential_18/dense_40/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_40/Tensordot/concat_1/axis°
)sequential_18/dense_40/Tensordot/concat_1ConcatV22sequential_18/dense_40/Tensordot/GatherV2:output:01sequential_18/dense_40/Tensordot/Const_2:output:07sequential_18/dense_40/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_18/dense_40/Tensordot/concat_1
 sequential_18/dense_40/TensordotReshape1sequential_18/dense_40/Tensordot/MatMul:product:02sequential_18/dense_40/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2"
 sequential_18/dense_40/TensordotÑ
-sequential_18/dense_40/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_40_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_18/dense_40/BiasAdd/ReadVariableOp
sequential_18/dense_40/BiasAddBiasAdd)sequential_18/dense_40/Tensordot:output:05sequential_18/dense_40/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2 
sequential_18/dense_40/BiasAdd
IdentityIdentity'sequential_18/dense_40/BiasAdd:output:0*
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
input_15

Ã
G__inference_sequential_18_layer_call_and_return_conditional_losses_5360
input_15
dense_38_5344
dense_38_5346
dense_39_5349
dense_39_5351
dense_40_5354
dense_40_5356
identity¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall´
 dense_38/StatefulPartitionedCallStatefulPartitionedCallinput_15dense_38_5344dense_38_5346*
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
B__inference_dense_38_layer_call_and_return_conditional_losses_52312"
 dense_38/StatefulPartitionedCallÕ
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_5349dense_39_5351*
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
B__inference_dense_39_layer_call_and_return_conditional_losses_52782"
 dense_39/StatefulPartitionedCallÕ
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_5354dense_40_5356*
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
B__inference_dense_40_layer_call_and_return_conditional_losses_53242"
 dense_40/StatefulPartitionedCall
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_15

ª
 __inference__traced_restore_5842
file_prefix$
 assignvariableop_dense_38_kernel$
 assignvariableop_1_dense_38_bias&
"assignvariableop_2_dense_39_kernel$
 assignvariableop_3_dense_39_bias&
"assignvariableop_4_dense_40_kernel$
 assignvariableop_5_dense_40_bias

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
AssignVariableOpAssignVariableOp assignvariableop_dense_38_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_38_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_39_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_39_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_40_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_40_biasIdentity_5:output:0"/device:CPU:0*
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
è
|
'__inference_dense_40_layer_call_fn_5773

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
B__inference_dense_40_layer_call_and_return_conditional_losses_53242
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

Ã
G__inference_sequential_18_layer_call_and_return_conditional_losses_5341
input_15
dense_38_5242
dense_38_5244
dense_39_5289
dense_39_5291
dense_40_5335
dense_40_5337
identity¢ dense_38/StatefulPartitionedCall¢ dense_39/StatefulPartitionedCall¢ dense_40/StatefulPartitionedCall´
 dense_38/StatefulPartitionedCallStatefulPartitionedCallinput_15dense_38_5242dense_38_5244*
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
B__inference_dense_38_layer_call_and_return_conditional_losses_52312"
 dense_38/StatefulPartitionedCallÕ
 dense_39/StatefulPartitionedCallStatefulPartitionedCall)dense_38/StatefulPartitionedCall:output:0dense_39_5289dense_39_5291*
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
B__inference_dense_39_layer_call_and_return_conditional_losses_52782"
 dense_39/StatefulPartitionedCallÕ
 dense_40/StatefulPartitionedCallStatefulPartitionedCall)dense_39/StatefulPartitionedCall:output:0dense_40_5335dense_40_5337*
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
B__inference_dense_40_layer_call_and_return_conditional_losses_53242"
 dense_40/StatefulPartitionedCall
IdentityIdentity)dense_40/StatefulPartitionedCall:output:0!^dense_38/StatefulPartitionedCall!^dense_39/StatefulPartitionedCall!^dense_40/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_38/StatefulPartitionedCall dense_38/StatefulPartitionedCall2D
 dense_39/StatefulPartitionedCall dense_39/StatefulPartitionedCall2D
 dense_40/StatefulPartitionedCall dense_40/StatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_15
à 
­
B__inference_dense_39_layer_call_and_return_conditional_losses_5725

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
input_15U
serving_default_input_15:05ÿÿÿÿÿÿÿÿÿ`
dense_40T
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
_tf_keras_sequentialÐ{"class_name": "Sequential", "name": "sequential_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_15"}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_15"}}, {"class_name": "Dense", "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_layerÌ{"class_name": "Dense", "name": "dense_38", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_38", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"class_name": "Dense", "name": "dense_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_39", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 32]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
7__call__
*8&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "Dense", "name": "dense_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_40", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 32]}}
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
!: 2dense_38/kernel
: 2dense_38/bias
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
!:  2dense_39/kernel
: 2dense_39/bias
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
!: 2dense_40/kernel
:2dense_40/bias
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
__inference__wrapped_model_5196Û
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
input_155ÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_sequential_18_layer_call_fn_5637
,__inference_sequential_18_layer_call_fn_5433
,__inference_sequential_18_layer_call_fn_5654
,__inference_sequential_18_layer_call_fn_5397À
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
G__inference_sequential_18_layer_call_and_return_conditional_losses_5536
G__inference_sequential_18_layer_call_and_return_conditional_losses_5620
G__inference_sequential_18_layer_call_and_return_conditional_losses_5341
G__inference_sequential_18_layer_call_and_return_conditional_losses_5360À
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
'__inference_dense_38_layer_call_fn_5694¢
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
B__inference_dense_38_layer_call_and_return_conditional_losses_5685¢
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
'__inference_dense_39_layer_call_fn_5734¢
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
B__inference_dense_39_layer_call_and_return_conditional_losses_5725¢
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
'__inference_dense_40_layer_call_fn_5773¢
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
B__inference_dense_40_layer_call_and_return_conditional_losses_5764¢
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
"__inference_signature_wrapper_5452input_15Ü
__inference__wrapped_model_5196¸
U¢R
K¢H
FC
input_155ÿÿÿÿÿÿÿÿÿ
ª "WªT
R
dense_40FC
dense_405ÿÿÿÿÿÿÿÿÿë
B__inference_dense_38_layer_call_and_return_conditional_losses_5685¤
S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ 
 Ã
'__inference_dense_38_layer_call_fn_5694
S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ
ª "<95ÿÿÿÿÿÿÿÿÿ ë
B__inference_dense_39_layer_call_and_return_conditional_losses_5725¤S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ 
 Ã
'__inference_dense_39_layer_call_fn_5734S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "<95ÿÿÿÿÿÿÿÿÿ ë
B__inference_dense_40_layer_call_and_return_conditional_losses_5764¤S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 Ã
'__inference_dense_40_layer_call_fn_5773S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "<95ÿÿÿÿÿÿÿÿÿþ
G__inference_sequential_18_layer_call_and_return_conditional_losses_5341²
]¢Z
S¢P
FC
input_155ÿÿÿÿÿÿÿÿÿ
p

 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 þ
G__inference_sequential_18_layer_call_and_return_conditional_losses_5360²
]¢Z
S¢P
FC
input_155ÿÿÿÿÿÿÿÿÿ
p 

 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 ü
G__inference_sequential_18_layer_call_and_return_conditional_losses_5536°
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
G__inference_sequential_18_layer_call_and_return_conditional_losses_5620°
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
,__inference_sequential_18_layer_call_fn_5397¥
]¢Z
S¢P
FC
input_155ÿÿÿÿÿÿÿÿÿ
p

 
ª "<95ÿÿÿÿÿÿÿÿÿÖ
,__inference_sequential_18_layer_call_fn_5433¥
]¢Z
S¢P
FC
input_155ÿÿÿÿÿÿÿÿÿ
p 

 
ª "<95ÿÿÿÿÿÿÿÿÿÔ
,__inference_sequential_18_layer_call_fn_5637£
[¢X
Q¢N
DA
inputs5ÿÿÿÿÿÿÿÿÿ
p

 
ª "<95ÿÿÿÿÿÿÿÿÿÔ
,__inference_sequential_18_layer_call_fn_5654£
[¢X
Q¢N
DA
inputs5ÿÿÿÿÿÿÿÿÿ
p 

 
ª "<95ÿÿÿÿÿÿÿÿÿë
"__inference_signature_wrapper_5452Ä
a¢^
¢ 
WªT
R
input_15FC
input_155ÿÿÿÿÿÿÿÿÿ"WªT
R
dense_40FC
dense_405ÿÿÿÿÿÿÿÿÿ