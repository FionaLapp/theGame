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
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

: *
dtype0
r
dense_32/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_32/bias
k
!dense_32/bias/Read/ReadVariableOpReadVariableOpdense_32/bias*
_output_shapes
: *
dtype0
z
dense_33/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  * 
shared_namedense_33/kernel
s
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes

:  *
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
_output_shapes
: *
dtype0
z
dense_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_34/kernel
s
#dense_34/kernel/Read/ReadVariableOpReadVariableOpdense_34/kernel*
_output_shapes

: *
dtype0
r
dense_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_34/bias
k
!dense_34/bias/Read/ReadVariableOpReadVariableOpdense_34/bias*
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
VARIABLE_VALUEdense_32/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_33/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_34/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_34/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_input_13Placeholder*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ*
dtype0*@
shape7:5ÿÿÿÿÿÿÿÿÿ
Á
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13dense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/bias*
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
"__inference_signature_wrapper_4332
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
÷
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOp#dense_34/kernel/Read/ReadVariableOp!dense_34/bias/Read/ReadVariableOpConst*
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
__inference__traced_save_4694
ú
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_32/kerneldense_32/biasdense_33/kerneldense_33/biasdense_34/kerneldense_34/bias*
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
 __inference__traced_restore_4722ëë
à
­
B__inference_dense_34_layer_call_and_return_conditional_losses_4644

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
'__inference_dense_32_layer_call_fn_4574

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
B__inference_dense_32_layer_call_and_return_conditional_losses_41112
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
ó
½
,__inference_sequential_16_layer_call_fn_4534

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
G__inference_sequential_16_layer_call_and_return_conditional_losses_42982
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
B__inference_dense_34_layer_call_and_return_conditional_losses_4204

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
G__inference_sequential_16_layer_call_and_return_conditional_losses_4262

inputs
dense_32_4246
dense_32_4248
dense_33_4251
dense_33_4253
dense_34_4256
dense_34_4258
identity¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall²
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_4246dense_32_4248*
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
B__inference_dense_32_layer_call_and_return_conditional_losses_41112"
 dense_32/StatefulPartitionedCallÕ
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_4251dense_33_4253*
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
B__inference_dense_33_layer_call_and_return_conditional_losses_41582"
 dense_33/StatefulPartitionedCallÕ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_4256dense_34_4258*
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
B__inference_dense_34_layer_call_and_return_conditional_losses_42042"
 dense_34/StatefulPartitionedCall
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
¿
,__inference_sequential_16_layer_call_fn_4313
input_13
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
G__inference_sequential_16_layer_call_and_return_conditional_losses_42982
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
input_13
è
|
'__inference_dense_34_layer_call_fn_4653

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
B__inference_dense_34_layer_call_and_return_conditional_losses_42042
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
à 
­
B__inference_dense_33_layer_call_and_return_conditional_losses_4158

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
Ç
µ
"__inference_signature_wrapper_4332
input_13
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
__inference__wrapped_model_40762
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
input_13
ÿk

G__inference_sequential_16_layer_call_and_return_conditional_losses_4416

inputs.
*dense_32_tensordot_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource.
*dense_33_tensordot_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource.
*dense_34_tensordot_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity±
!dense_32/Tensordot/ReadVariableOpReadVariableOp*dense_32_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_32/Tensordot/ReadVariableOp|
dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_32/Tensordot/axes£
dense_32/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_32/Tensordot/freej
dense_32/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_32/Tensordot/Shape
 dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/GatherV2/axisþ
dense_32/Tensordot/GatherV2GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/free:output:0)dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_32/Tensordot/GatherV2
"dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_32/Tensordot/GatherV2_1/axis
dense_32/Tensordot/GatherV2_1GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/axes:output:0+dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2_1~
dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const¤
dense_32/Tensordot/ProdProd$dense_32/Tensordot/GatherV2:output:0!dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod
dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_1¬
dense_32/Tensordot/Prod_1Prod&dense_32/Tensordot/GatherV2_1:output:0#dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod_1
dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_32/Tensordot/concat/axisÝ
dense_32/Tensordot/concatConcatV2 dense_32/Tensordot/free:output:0 dense_32/Tensordot/axes:output:0'dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat°
dense_32/Tensordot/stackPack dense_32/Tensordot/Prod:output:0"dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/stackË
dense_32/Tensordot/transpose	Transposeinputs"dense_32/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot/transposeÃ
dense_32/Tensordot/ReshapeReshape dense_32/Tensordot/transpose:y:0!dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot/ReshapeÂ
dense_32/Tensordot/MatMulMatMul#dense_32/Tensordot/Reshape:output:0)dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_32/Tensordot/MatMul
dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_2
 dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/concat_1/axisê
dense_32/Tensordot/concat_1ConcatV2$dense_32/Tensordot/GatherV2:output:0#dense_32/Tensordot/Const_2:output:0)dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat_1Ô
dense_32/TensordotReshape#dense_32/Tensordot/MatMul:product:0$dense_32/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_32/Tensordot§
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_32/BiasAdd/ReadVariableOpË
dense_32/BiasAddBiasAdddense_32/Tensordot:output:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_32/BiasAdd
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_32/Relu±
!dense_33/Tensordot/ReadVariableOpReadVariableOp*dense_33_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_33/Tensordot/ReadVariableOp|
dense_33/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_33/Tensordot/axes£
dense_33/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_33/Tensordot/free
dense_33/Tensordot/ShapeShapedense_32/Relu:activations:0*
T0*
_output_shapes
:2
dense_33/Tensordot/Shape
 dense_33/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_33/Tensordot/GatherV2/axisþ
dense_33/Tensordot/GatherV2GatherV2!dense_33/Tensordot/Shape:output:0 dense_33/Tensordot/free:output:0)dense_33/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_33/Tensordot/GatherV2
"dense_33/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_33/Tensordot/GatherV2_1/axis
dense_33/Tensordot/GatherV2_1GatherV2!dense_33/Tensordot/Shape:output:0 dense_33/Tensordot/axes:output:0+dense_33/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_33/Tensordot/GatherV2_1~
dense_33/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_33/Tensordot/Const¤
dense_33/Tensordot/ProdProd$dense_33/Tensordot/GatherV2:output:0!dense_33/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_33/Tensordot/Prod
dense_33/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_33/Tensordot/Const_1¬
dense_33/Tensordot/Prod_1Prod&dense_33/Tensordot/GatherV2_1:output:0#dense_33/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_33/Tensordot/Prod_1
dense_33/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_33/Tensordot/concat/axisÝ
dense_33/Tensordot/concatConcatV2 dense_33/Tensordot/free:output:0 dense_33/Tensordot/axes:output:0'dense_33/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/concat°
dense_33/Tensordot/stackPack dense_33/Tensordot/Prod:output:0"dense_33/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/stackà
dense_33/Tensordot/transpose	Transposedense_32/Relu:activations:0"dense_33/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_33/Tensordot/transposeÃ
dense_33/Tensordot/ReshapeReshape dense_33/Tensordot/transpose:y:0!dense_33/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_33/Tensordot/ReshapeÂ
dense_33/Tensordot/MatMulMatMul#dense_33/Tensordot/Reshape:output:0)dense_33/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_33/Tensordot/MatMul
dense_33/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_33/Tensordot/Const_2
 dense_33/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_33/Tensordot/concat_1/axisê
dense_33/Tensordot/concat_1ConcatV2$dense_33/Tensordot/GatherV2:output:0#dense_33/Tensordot/Const_2:output:0)dense_33/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/concat_1Ô
dense_33/TensordotReshape#dense_33/Tensordot/MatMul:product:0$dense_33/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_33/Tensordot§
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_33/BiasAdd/ReadVariableOpË
dense_33/BiasAddBiasAdddense_33/Tensordot:output:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_33/BiasAdd
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_33/Relu±
!dense_34/Tensordot/ReadVariableOpReadVariableOp*dense_34_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_34/Tensordot/ReadVariableOp|
dense_34/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_34/Tensordot/axes£
dense_34/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_34/Tensordot/free
dense_34/Tensordot/ShapeShapedense_33/Relu:activations:0*
T0*
_output_shapes
:2
dense_34/Tensordot/Shape
 dense_34/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_34/Tensordot/GatherV2/axisþ
dense_34/Tensordot/GatherV2GatherV2!dense_34/Tensordot/Shape:output:0 dense_34/Tensordot/free:output:0)dense_34/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_34/Tensordot/GatherV2
"dense_34/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_34/Tensordot/GatherV2_1/axis
dense_34/Tensordot/GatherV2_1GatherV2!dense_34/Tensordot/Shape:output:0 dense_34/Tensordot/axes:output:0+dense_34/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_34/Tensordot/GatherV2_1~
dense_34/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_34/Tensordot/Const¤
dense_34/Tensordot/ProdProd$dense_34/Tensordot/GatherV2:output:0!dense_34/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_34/Tensordot/Prod
dense_34/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_34/Tensordot/Const_1¬
dense_34/Tensordot/Prod_1Prod&dense_34/Tensordot/GatherV2_1:output:0#dense_34/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/Tensordot/Prod_1
dense_34/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_34/Tensordot/concat/axisÝ
dense_34/Tensordot/concatConcatV2 dense_34/Tensordot/free:output:0 dense_34/Tensordot/axes:output:0'dense_34/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_34/Tensordot/concat°
dense_34/Tensordot/stackPack dense_34/Tensordot/Prod:output:0"dense_34/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_34/Tensordot/stackà
dense_34/Tensordot/transpose	Transposedense_33/Relu:activations:0"dense_34/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_34/Tensordot/transposeÃ
dense_34/Tensordot/ReshapeReshape dense_34/Tensordot/transpose:y:0!dense_34/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_34/Tensordot/ReshapeÂ
dense_34/Tensordot/MatMulMatMul#dense_34/Tensordot/Reshape:output:0)dense_34/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_34/Tensordot/MatMul
dense_34/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_34/Tensordot/Const_2
 dense_34/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_34/Tensordot/concat_1/axisê
dense_34/Tensordot/concat_1ConcatV2$dense_34/Tensordot/GatherV2:output:0#dense_34/Tensordot/Const_2:output:0)dense_34/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_34/Tensordot/concat_1Ô
dense_34/TensordotReshape#dense_34/Tensordot/MatMul:product:0$dense_34/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_34/Tensordot§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOpË
dense_34/BiasAddBiasAdddense_34/Tensordot:output:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_34/BiasAdd
IdentityIdentitydense_34/BiasAdd:output:0*
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

ª
 __inference__traced_restore_4722
file_prefix$
 assignvariableop_dense_32_kernel$
 assignvariableop_1_dense_32_bias&
"assignvariableop_2_dense_33_kernel$
 assignvariableop_3_dense_33_bias&
"assignvariableop_4_dense_34_kernel$
 assignvariableop_5_dense_34_bias

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
AssignVariableOpAssignVariableOp assignvariableop_dense_32_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_32_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_33_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_33_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_34_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_34_biasIdentity_5:output:0"/device:CPU:0*
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

Ã
G__inference_sequential_16_layer_call_and_return_conditional_losses_4240
input_13
dense_32_4224
dense_32_4226
dense_33_4229
dense_33_4231
dense_34_4234
dense_34_4236
identity¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall´
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_32_4224dense_32_4226*
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
B__inference_dense_32_layer_call_and_return_conditional_losses_41112"
 dense_32/StatefulPartitionedCallÕ
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_4229dense_33_4231*
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
B__inference_dense_33_layer_call_and_return_conditional_losses_41582"
 dense_33/StatefulPartitionedCallÕ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_4234dense_34_4236*
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
B__inference_dense_34_layer_call_and_return_conditional_losses_42042"
 dense_34/StatefulPartitionedCall
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_13

Á
G__inference_sequential_16_layer_call_and_return_conditional_losses_4298

inputs
dense_32_4282
dense_32_4284
dense_33_4287
dense_33_4289
dense_34_4292
dense_34_4294
identity¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall²
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_4282dense_32_4284*
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
B__inference_dense_32_layer_call_and_return_conditional_losses_41112"
 dense_32/StatefulPartitionedCallÕ
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_4287dense_33_4289*
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
B__inference_dense_33_layer_call_and_return_conditional_losses_41582"
 dense_33/StatefulPartitionedCallÕ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_4292dense_34_4294*
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
B__inference_dense_34_layer_call_and_return_conditional_losses_42042"
 dense_34/StatefulPartitionedCall
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:s o
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
¿
,__inference_sequential_16_layer_call_fn_4277
input_13
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÑ
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
G__inference_sequential_16_layer_call_and_return_conditional_losses_42622
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
input_13

Ã
G__inference_sequential_16_layer_call_and_return_conditional_losses_4221
input_13
dense_32_4122
dense_32_4124
dense_33_4169
dense_33_4171
dense_34_4215
dense_34_4217
identity¢ dense_32/StatefulPartitionedCall¢ dense_33/StatefulPartitionedCall¢ dense_34/StatefulPartitionedCall´
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_32_4122dense_32_4124*
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
B__inference_dense_32_layer_call_and_return_conditional_losses_41112"
 dense_32/StatefulPartitionedCallÕ
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_4169dense_33_4171*
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
B__inference_dense_33_layer_call_and_return_conditional_losses_41582"
 dense_33/StatefulPartitionedCallÕ
 dense_34/StatefulPartitionedCallStatefulPartitionedCall)dense_33/StatefulPartitionedCall:output:0dense_34_4215dense_34_4217*
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
B__inference_dense_34_layer_call_and_return_conditional_losses_42042"
 dense_34/StatefulPartitionedCall
IdentityIdentity)dense_34/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall!^dense_34/StatefulPartitionedCall*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*b
_input_shapesQ
O:5ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall2D
 dense_34/StatefulPartitionedCall dense_34/StatefulPartitionedCall:u q
K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
input_13
à 
­
B__inference_dense_32_layer_call_and_return_conditional_losses_4565

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
'__inference_dense_33_layer_call_fn_4614

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
B__inference_dense_33_layer_call_and_return_conditional_losses_41582
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
à 
­
B__inference_dense_33_layer_call_and_return_conditional_losses_4605

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
ø
®
__inference__wrapped_model_4076
input_13<
8sequential_16_dense_32_tensordot_readvariableop_resource:
6sequential_16_dense_32_biasadd_readvariableop_resource<
8sequential_16_dense_33_tensordot_readvariableop_resource:
6sequential_16_dense_33_biasadd_readvariableop_resource<
8sequential_16_dense_34_tensordot_readvariableop_resource:
6sequential_16_dense_34_biasadd_readvariableop_resource
identityÛ
/sequential_16/dense_32/Tensordot/ReadVariableOpReadVariableOp8sequential_16_dense_32_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_16/dense_32/Tensordot/ReadVariableOp
%sequential_16/dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_16/dense_32/Tensordot/axes¿
%sequential_16/dense_32/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_16/dense_32/Tensordot/free
&sequential_16/dense_32/Tensordot/ShapeShapeinput_13*
T0*
_output_shapes
:2(
&sequential_16/dense_32/Tensordot/Shape¢
.sequential_16/dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_32/Tensordot/GatherV2/axisÄ
)sequential_16/dense_32/Tensordot/GatherV2GatherV2/sequential_16/dense_32/Tensordot/Shape:output:0.sequential_16/dense_32/Tensordot/free:output:07sequential_16/dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_16/dense_32/Tensordot/GatherV2¦
0sequential_16/dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_16/dense_32/Tensordot/GatherV2_1/axisÊ
+sequential_16/dense_32/Tensordot/GatherV2_1GatherV2/sequential_16/dense_32/Tensordot/Shape:output:0.sequential_16/dense_32/Tensordot/axes:output:09sequential_16/dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_16/dense_32/Tensordot/GatherV2_1
&sequential_16/dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_16/dense_32/Tensordot/ConstÜ
%sequential_16/dense_32/Tensordot/ProdProd2sequential_16/dense_32/Tensordot/GatherV2:output:0/sequential_16/dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_16/dense_32/Tensordot/Prod
(sequential_16/dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_32/Tensordot/Const_1ä
'sequential_16/dense_32/Tensordot/Prod_1Prod4sequential_16/dense_32/Tensordot/GatherV2_1:output:01sequential_16/dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_16/dense_32/Tensordot/Prod_1
,sequential_16/dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_16/dense_32/Tensordot/concat/axis£
'sequential_16/dense_32/Tensordot/concatConcatV2.sequential_16/dense_32/Tensordot/free:output:0.sequential_16/dense_32/Tensordot/axes:output:05sequential_16/dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_16/dense_32/Tensordot/concatè
&sequential_16/dense_32/Tensordot/stackPack.sequential_16/dense_32/Tensordot/Prod:output:00sequential_16/dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_16/dense_32/Tensordot/stack÷
*sequential_16/dense_32/Tensordot/transpose	Transposeinput_130sequential_16/dense_32/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2,
*sequential_16/dense_32/Tensordot/transposeû
(sequential_16/dense_32/Tensordot/ReshapeReshape.sequential_16/dense_32/Tensordot/transpose:y:0/sequential_16/dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_16/dense_32/Tensordot/Reshapeú
'sequential_16/dense_32/Tensordot/MatMulMatMul1sequential_16/dense_32/Tensordot/Reshape:output:07sequential_16/dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_16/dense_32/Tensordot/MatMul
(sequential_16/dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_32/Tensordot/Const_2¢
.sequential_16/dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_32/Tensordot/concat_1/axis°
)sequential_16/dense_32/Tensordot/concat_1ConcatV22sequential_16/dense_32/Tensordot/GatherV2:output:01sequential_16/dense_32/Tensordot/Const_2:output:07sequential_16/dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_16/dense_32/Tensordot/concat_1
 sequential_16/dense_32/TensordotReshape1sequential_16/dense_32/Tensordot/MatMul:product:02sequential_16/dense_32/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2"
 sequential_16/dense_32/TensordotÑ
-sequential_16/dense_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_16/dense_32/BiasAdd/ReadVariableOp
sequential_16/dense_32/BiasAddBiasAdd)sequential_16/dense_32/Tensordot:output:05sequential_16/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2 
sequential_16/dense_32/BiasAddÁ
sequential_16/dense_32/ReluRelu'sequential_16/dense_32/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
sequential_16/dense_32/ReluÛ
/sequential_16/dense_33/Tensordot/ReadVariableOpReadVariableOp8sequential_16_dense_33_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype021
/sequential_16/dense_33/Tensordot/ReadVariableOp
%sequential_16/dense_33/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_16/dense_33/Tensordot/axes¿
%sequential_16/dense_33/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_16/dense_33/Tensordot/free©
&sequential_16/dense_33/Tensordot/ShapeShape)sequential_16/dense_32/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_16/dense_33/Tensordot/Shape¢
.sequential_16/dense_33/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_33/Tensordot/GatherV2/axisÄ
)sequential_16/dense_33/Tensordot/GatherV2GatherV2/sequential_16/dense_33/Tensordot/Shape:output:0.sequential_16/dense_33/Tensordot/free:output:07sequential_16/dense_33/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_16/dense_33/Tensordot/GatherV2¦
0sequential_16/dense_33/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_16/dense_33/Tensordot/GatherV2_1/axisÊ
+sequential_16/dense_33/Tensordot/GatherV2_1GatherV2/sequential_16/dense_33/Tensordot/Shape:output:0.sequential_16/dense_33/Tensordot/axes:output:09sequential_16/dense_33/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_16/dense_33/Tensordot/GatherV2_1
&sequential_16/dense_33/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_16/dense_33/Tensordot/ConstÜ
%sequential_16/dense_33/Tensordot/ProdProd2sequential_16/dense_33/Tensordot/GatherV2:output:0/sequential_16/dense_33/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_16/dense_33/Tensordot/Prod
(sequential_16/dense_33/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_33/Tensordot/Const_1ä
'sequential_16/dense_33/Tensordot/Prod_1Prod4sequential_16/dense_33/Tensordot/GatherV2_1:output:01sequential_16/dense_33/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_16/dense_33/Tensordot/Prod_1
,sequential_16/dense_33/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_16/dense_33/Tensordot/concat/axis£
'sequential_16/dense_33/Tensordot/concatConcatV2.sequential_16/dense_33/Tensordot/free:output:0.sequential_16/dense_33/Tensordot/axes:output:05sequential_16/dense_33/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_16/dense_33/Tensordot/concatè
&sequential_16/dense_33/Tensordot/stackPack.sequential_16/dense_33/Tensordot/Prod:output:00sequential_16/dense_33/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_16/dense_33/Tensordot/stack
*sequential_16/dense_33/Tensordot/transpose	Transpose)sequential_16/dense_32/Relu:activations:00sequential_16/dense_33/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2,
*sequential_16/dense_33/Tensordot/transposeû
(sequential_16/dense_33/Tensordot/ReshapeReshape.sequential_16/dense_33/Tensordot/transpose:y:0/sequential_16/dense_33/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_16/dense_33/Tensordot/Reshapeú
'sequential_16/dense_33/Tensordot/MatMulMatMul1sequential_16/dense_33/Tensordot/Reshape:output:07sequential_16/dense_33/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2)
'sequential_16/dense_33/Tensordot/MatMul
(sequential_16/dense_33/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_33/Tensordot/Const_2¢
.sequential_16/dense_33/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_33/Tensordot/concat_1/axis°
)sequential_16/dense_33/Tensordot/concat_1ConcatV22sequential_16/dense_33/Tensordot/GatherV2:output:01sequential_16/dense_33/Tensordot/Const_2:output:07sequential_16/dense_33/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_16/dense_33/Tensordot/concat_1
 sequential_16/dense_33/TensordotReshape1sequential_16/dense_33/Tensordot/MatMul:product:02sequential_16/dense_33/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2"
 sequential_16/dense_33/TensordotÑ
-sequential_16/dense_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_16/dense_33/BiasAdd/ReadVariableOp
sequential_16/dense_33/BiasAddBiasAdd)sequential_16/dense_33/Tensordot:output:05sequential_16/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2 
sequential_16/dense_33/BiasAddÁ
sequential_16/dense_33/ReluRelu'sequential_16/dense_33/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
sequential_16/dense_33/ReluÛ
/sequential_16/dense_34/Tensordot/ReadVariableOpReadVariableOp8sequential_16_dense_34_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_16/dense_34/Tensordot/ReadVariableOp
%sequential_16/dense_34/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2'
%sequential_16/dense_34/Tensordot/axes¿
%sequential_16/dense_34/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2'
%sequential_16/dense_34/Tensordot/free©
&sequential_16/dense_34/Tensordot/ShapeShape)sequential_16/dense_33/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_16/dense_34/Tensordot/Shape¢
.sequential_16/dense_34/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_34/Tensordot/GatherV2/axisÄ
)sequential_16/dense_34/Tensordot/GatherV2GatherV2/sequential_16/dense_34/Tensordot/Shape:output:0.sequential_16/dense_34/Tensordot/free:output:07sequential_16/dense_34/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2+
)sequential_16/dense_34/Tensordot/GatherV2¦
0sequential_16/dense_34/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_16/dense_34/Tensordot/GatherV2_1/axisÊ
+sequential_16/dense_34/Tensordot/GatherV2_1GatherV2/sequential_16/dense_34/Tensordot/Shape:output:0.sequential_16/dense_34/Tensordot/axes:output:09sequential_16/dense_34/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_16/dense_34/Tensordot/GatherV2_1
&sequential_16/dense_34/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_16/dense_34/Tensordot/ConstÜ
%sequential_16/dense_34/Tensordot/ProdProd2sequential_16/dense_34/Tensordot/GatherV2:output:0/sequential_16/dense_34/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_16/dense_34/Tensordot/Prod
(sequential_16/dense_34/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_34/Tensordot/Const_1ä
'sequential_16/dense_34/Tensordot/Prod_1Prod4sequential_16/dense_34/Tensordot/GatherV2_1:output:01sequential_16/dense_34/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_16/dense_34/Tensordot/Prod_1
,sequential_16/dense_34/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_16/dense_34/Tensordot/concat/axis£
'sequential_16/dense_34/Tensordot/concatConcatV2.sequential_16/dense_34/Tensordot/free:output:0.sequential_16/dense_34/Tensordot/axes:output:05sequential_16/dense_34/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_16/dense_34/Tensordot/concatè
&sequential_16/dense_34/Tensordot/stackPack.sequential_16/dense_34/Tensordot/Prod:output:00sequential_16/dense_34/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_16/dense_34/Tensordot/stack
*sequential_16/dense_34/Tensordot/transpose	Transpose)sequential_16/dense_33/Relu:activations:00sequential_16/dense_34/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2,
*sequential_16/dense_34/Tensordot/transposeû
(sequential_16/dense_34/Tensordot/ReshapeReshape.sequential_16/dense_34/Tensordot/transpose:y:0/sequential_16/dense_34/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2*
(sequential_16/dense_34/Tensordot/Reshapeú
'sequential_16/dense_34/Tensordot/MatMulMatMul1sequential_16/dense_34/Tensordot/Reshape:output:07sequential_16/dense_34/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_16/dense_34/Tensordot/MatMul
(sequential_16/dense_34/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_16/dense_34/Tensordot/Const_2¢
.sequential_16/dense_34/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_34/Tensordot/concat_1/axis°
)sequential_16/dense_34/Tensordot/concat_1ConcatV22sequential_16/dense_34/Tensordot/GatherV2:output:01sequential_16/dense_34/Tensordot/Const_2:output:07sequential_16/dense_34/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_16/dense_34/Tensordot/concat_1
 sequential_16/dense_34/TensordotReshape1sequential_16/dense_34/Tensordot/MatMul:product:02sequential_16/dense_34/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2"
 sequential_16/dense_34/TensordotÑ
-sequential_16/dense_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_16/dense_34/BiasAdd/ReadVariableOp
sequential_16/dense_34/BiasAddBiasAdd)sequential_16/dense_34/Tensordot:output:05sequential_16/dense_34/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2 
sequential_16/dense_34/BiasAdd
IdentityIdentity'sequential_16/dense_34/BiasAdd:output:0*
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
input_13
ó
½
,__inference_sequential_16_layer_call_fn_4517

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
G__inference_sequential_16_layer_call_and_return_conditional_losses_42622
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
ÿk

G__inference_sequential_16_layer_call_and_return_conditional_losses_4500

inputs.
*dense_32_tensordot_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource.
*dense_33_tensordot_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource.
*dense_34_tensordot_readvariableop_resource,
(dense_34_biasadd_readvariableop_resource
identity±
!dense_32/Tensordot/ReadVariableOpReadVariableOp*dense_32_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_32/Tensordot/ReadVariableOp|
dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_32/Tensordot/axes£
dense_32/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_32/Tensordot/freej
dense_32/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_32/Tensordot/Shape
 dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/GatherV2/axisþ
dense_32/Tensordot/GatherV2GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/free:output:0)dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_32/Tensordot/GatherV2
"dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_32/Tensordot/GatherV2_1/axis
dense_32/Tensordot/GatherV2_1GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/axes:output:0+dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2_1~
dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const¤
dense_32/Tensordot/ProdProd$dense_32/Tensordot/GatherV2:output:0!dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod
dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_1¬
dense_32/Tensordot/Prod_1Prod&dense_32/Tensordot/GatherV2_1:output:0#dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod_1
dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_32/Tensordot/concat/axisÝ
dense_32/Tensordot/concatConcatV2 dense_32/Tensordot/free:output:0 dense_32/Tensordot/axes:output:0'dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat°
dense_32/Tensordot/stackPack dense_32/Tensordot/Prod:output:0"dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/stackË
dense_32/Tensordot/transpose	Transposeinputs"dense_32/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot/transposeÃ
dense_32/Tensordot/ReshapeReshape dense_32/Tensordot/transpose:y:0!dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_32/Tensordot/ReshapeÂ
dense_32/Tensordot/MatMulMatMul#dense_32/Tensordot/Reshape:output:0)dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_32/Tensordot/MatMul
dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_2
 dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/concat_1/axisê
dense_32/Tensordot/concat_1ConcatV2$dense_32/Tensordot/GatherV2:output:0#dense_32/Tensordot/Const_2:output:0)dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat_1Ô
dense_32/TensordotReshape#dense_32/Tensordot/MatMul:product:0$dense_32/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_32/Tensordot§
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_32/BiasAdd/ReadVariableOpË
dense_32/BiasAddBiasAdddense_32/Tensordot:output:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_32/BiasAdd
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_32/Relu±
!dense_33/Tensordot/ReadVariableOpReadVariableOp*dense_33_tensordot_readvariableop_resource*
_output_shapes

:  *
dtype02#
!dense_33/Tensordot/ReadVariableOp|
dense_33/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_33/Tensordot/axes£
dense_33/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_33/Tensordot/free
dense_33/Tensordot/ShapeShapedense_32/Relu:activations:0*
T0*
_output_shapes
:2
dense_33/Tensordot/Shape
 dense_33/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_33/Tensordot/GatherV2/axisþ
dense_33/Tensordot/GatherV2GatherV2!dense_33/Tensordot/Shape:output:0 dense_33/Tensordot/free:output:0)dense_33/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_33/Tensordot/GatherV2
"dense_33/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_33/Tensordot/GatherV2_1/axis
dense_33/Tensordot/GatherV2_1GatherV2!dense_33/Tensordot/Shape:output:0 dense_33/Tensordot/axes:output:0+dense_33/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_33/Tensordot/GatherV2_1~
dense_33/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_33/Tensordot/Const¤
dense_33/Tensordot/ProdProd$dense_33/Tensordot/GatherV2:output:0!dense_33/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_33/Tensordot/Prod
dense_33/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_33/Tensordot/Const_1¬
dense_33/Tensordot/Prod_1Prod&dense_33/Tensordot/GatherV2_1:output:0#dense_33/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_33/Tensordot/Prod_1
dense_33/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_33/Tensordot/concat/axisÝ
dense_33/Tensordot/concatConcatV2 dense_33/Tensordot/free:output:0 dense_33/Tensordot/axes:output:0'dense_33/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/concat°
dense_33/Tensordot/stackPack dense_33/Tensordot/Prod:output:0"dense_33/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/stackà
dense_33/Tensordot/transpose	Transposedense_32/Relu:activations:0"dense_33/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_33/Tensordot/transposeÃ
dense_33/Tensordot/ReshapeReshape dense_33/Tensordot/transpose:y:0!dense_33/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_33/Tensordot/ReshapeÂ
dense_33/Tensordot/MatMulMatMul#dense_33/Tensordot/Reshape:output:0)dense_33/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
dense_33/Tensordot/MatMul
dense_33/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_33/Tensordot/Const_2
 dense_33/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_33/Tensordot/concat_1/axisê
dense_33/Tensordot/concat_1ConcatV2$dense_33/Tensordot/GatherV2:output:0#dense_33/Tensordot/Const_2:output:0)dense_33/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/concat_1Ô
dense_33/TensordotReshape#dense_33/Tensordot/MatMul:product:0$dense_33/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_33/Tensordot§
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_33/BiasAdd/ReadVariableOpË
dense_33/BiasAddBiasAdddense_33/Tensordot:output:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_33/BiasAdd
dense_33/ReluReludense_33/BiasAdd:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_33/Relu±
!dense_34/Tensordot/ReadVariableOpReadVariableOp*dense_34_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_34/Tensordot/ReadVariableOp|
dense_34/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
2
dense_34/Tensordot/axes£
dense_34/Tensordot/freeConst*
_output_shapes
:
*
dtype0*=
value4B2
"(                            	   2
dense_34/Tensordot/free
dense_34/Tensordot/ShapeShapedense_33/Relu:activations:0*
T0*
_output_shapes
:2
dense_34/Tensordot/Shape
 dense_34/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_34/Tensordot/GatherV2/axisþ
dense_34/Tensordot/GatherV2GatherV2!dense_34/Tensordot/Shape:output:0 dense_34/Tensordot/free:output:0)dense_34/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
2
dense_34/Tensordot/GatherV2
"dense_34/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_34/Tensordot/GatherV2_1/axis
dense_34/Tensordot/GatherV2_1GatherV2!dense_34/Tensordot/Shape:output:0 dense_34/Tensordot/axes:output:0+dense_34/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_34/Tensordot/GatherV2_1~
dense_34/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_34/Tensordot/Const¤
dense_34/Tensordot/ProdProd$dense_34/Tensordot/GatherV2:output:0!dense_34/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_34/Tensordot/Prod
dense_34/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_34/Tensordot/Const_1¬
dense_34/Tensordot/Prod_1Prod&dense_34/Tensordot/GatherV2_1:output:0#dense_34/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_34/Tensordot/Prod_1
dense_34/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_34/Tensordot/concat/axisÝ
dense_34/Tensordot/concatConcatV2 dense_34/Tensordot/free:output:0 dense_34/Tensordot/axes:output:0'dense_34/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_34/Tensordot/concat°
dense_34/Tensordot/stackPack dense_34/Tensordot/Prod:output:0"dense_34/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_34/Tensordot/stackà
dense_34/Tensordot/transpose	Transposedense_33/Relu:activations:0"dense_34/Tensordot/concat:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ 2
dense_34/Tensordot/transposeÃ
dense_34/Tensordot/ReshapeReshape dense_34/Tensordot/transpose:y:0!dense_34/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_34/Tensordot/ReshapeÂ
dense_34/Tensordot/MatMulMatMul#dense_34/Tensordot/Reshape:output:0)dense_34/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_34/Tensordot/MatMul
dense_34/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_34/Tensordot/Const_2
 dense_34/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_34/Tensordot/concat_1/axisê
dense_34/Tensordot/concat_1ConcatV2$dense_34/Tensordot/GatherV2:output:0#dense_34/Tensordot/Const_2:output:0)dense_34/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_34/Tensordot/concat_1Ô
dense_34/TensordotReshape#dense_34/Tensordot/MatMul:product:0$dense_34/Tensordot/concat_1:output:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_34/Tensordot§
dense_34/BiasAdd/ReadVariableOpReadVariableOp(dense_34_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_34/BiasAdd/ReadVariableOpË
dense_34/BiasAddBiasAdddense_34/Tensordot:output:0'dense_34/BiasAdd/ReadVariableOp:value:0*
T0*K
_output_shapes9
7:5ÿÿÿÿÿÿÿÿÿ2
dense_34/BiasAdd
IdentityIdentitydense_34/BiasAdd:output:0*
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
B__inference_dense_32_layer_call_and_return_conditional_losses_4111

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
ù

__inference__traced_save_4694
file_prefix.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop.
*savev2_dense_34_kernel_read_readvariableop,
(savev2_dense_34_bias_read_readvariableop
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
value3B1 B+_temp_8626b0c28ea94ea589fd54398591ef94/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop*savev2_dense_34_kernel_read_readvariableop(savev2_dense_34_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
: "¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*õ
serving_defaultá
a
input_13U
serving_default_input_13:05ÿÿÿÿÿÿÿÿÿ`
dense_34T
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
_tf_keras_sequentialÐ{"class_name": "Sequential", "name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_layerÌ{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
5__call__
*6&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 32]}}


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
7__call__
*8&call_and_return_all_conditional_losses"é
_tf_keras_layerÏ{"class_name": "Dense", "name": "dense_34", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_34", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3, 3, 3, 3, 3, 3, 3, 3, 3, 32]}}
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
!: 2dense_32/kernel
: 2dense_32/bias
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
!:  2dense_33/kernel
: 2dense_33/bias
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
!: 2dense_34/kernel
:2dense_34/bias
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
__inference__wrapped_model_4076Û
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
input_135ÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_sequential_16_layer_call_fn_4277
,__inference_sequential_16_layer_call_fn_4313
,__inference_sequential_16_layer_call_fn_4534
,__inference_sequential_16_layer_call_fn_4517À
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
G__inference_sequential_16_layer_call_and_return_conditional_losses_4221
G__inference_sequential_16_layer_call_and_return_conditional_losses_4416
G__inference_sequential_16_layer_call_and_return_conditional_losses_4500
G__inference_sequential_16_layer_call_and_return_conditional_losses_4240À
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
'__inference_dense_32_layer_call_fn_4574¢
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
B__inference_dense_32_layer_call_and_return_conditional_losses_4565¢
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
'__inference_dense_33_layer_call_fn_4614¢
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
B__inference_dense_33_layer_call_and_return_conditional_losses_4605¢
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
'__inference_dense_34_layer_call_fn_4653¢
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
B__inference_dense_34_layer_call_and_return_conditional_losses_4644¢
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
"__inference_signature_wrapper_4332input_13Ü
__inference__wrapped_model_4076¸
U¢R
K¢H
FC
input_135ÿÿÿÿÿÿÿÿÿ
ª "WªT
R
dense_34FC
dense_345ÿÿÿÿÿÿÿÿÿë
B__inference_dense_32_layer_call_and_return_conditional_losses_4565¤
S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ 
 Ã
'__inference_dense_32_layer_call_fn_4574
S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ
ª "<95ÿÿÿÿÿÿÿÿÿ ë
B__inference_dense_33_layer_call_and_return_conditional_losses_4605¤S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ 
 Ã
'__inference_dense_33_layer_call_fn_4614S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "<95ÿÿÿÿÿÿÿÿÿ ë
B__inference_dense_34_layer_call_and_return_conditional_losses_4644¤S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 Ã
'__inference_dense_34_layer_call_fn_4653S¢P
I¢F
DA
inputs5ÿÿÿÿÿÿÿÿÿ 
ª "<95ÿÿÿÿÿÿÿÿÿþ
G__inference_sequential_16_layer_call_and_return_conditional_losses_4221²
]¢Z
S¢P
FC
input_135ÿÿÿÿÿÿÿÿÿ
p

 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 þ
G__inference_sequential_16_layer_call_and_return_conditional_losses_4240²
]¢Z
S¢P
FC
input_135ÿÿÿÿÿÿÿÿÿ
p 

 
ª "I¢F
?<
05ÿÿÿÿÿÿÿÿÿ
 ü
G__inference_sequential_16_layer_call_and_return_conditional_losses_4416°
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
G__inference_sequential_16_layer_call_and_return_conditional_losses_4500°
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
,__inference_sequential_16_layer_call_fn_4277¥
]¢Z
S¢P
FC
input_135ÿÿÿÿÿÿÿÿÿ
p

 
ª "<95ÿÿÿÿÿÿÿÿÿÖ
,__inference_sequential_16_layer_call_fn_4313¥
]¢Z
S¢P
FC
input_135ÿÿÿÿÿÿÿÿÿ
p 

 
ª "<95ÿÿÿÿÿÿÿÿÿÔ
,__inference_sequential_16_layer_call_fn_4517£
[¢X
Q¢N
DA
inputs5ÿÿÿÿÿÿÿÿÿ
p

 
ª "<95ÿÿÿÿÿÿÿÿÿÔ
,__inference_sequential_16_layer_call_fn_4534£
[¢X
Q¢N
DA
inputs5ÿÿÿÿÿÿÿÿÿ
p 

 
ª "<95ÿÿÿÿÿÿÿÿÿë
"__inference_signature_wrapper_4332Ä
a¢^
¢ 
WªT
R
input_13FC
input_135ÿÿÿÿÿÿÿÿÿ"WªT
R
dense_34FC
dense_345ÿÿÿÿÿÿÿÿÿ