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
 ?"serve*2.3.02unknown8ސ
z
dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_24/kernel
s
#dense_24/kernel/Read/ReadVariableOpReadVariableOpdense_24/kernel*
_output_shapes

: *
dtype0
r
dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_24/bias
k
!dense_24/bias/Read/ReadVariableOpReadVariableOpdense_24/bias*
_output_shapes
: *
dtype0
z
dense_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_25/kernel
s
#dense_25/kernel/Read/ReadVariableOpReadVariableOpdense_25/kernel*
_output_shapes

: *
dtype0
r
dense_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_25/bias
k
!dense_25/bias/Read/ReadVariableOpReadVariableOpdense_25/bias*
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
VARIABLE_VALUEdense_24/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_24/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_25/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_25/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_input_13Placeholder*+
_output_shapes
:?????????2*
dtype0* 
shape:?????????2
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_13dense_24/kerneldense_24/biasdense_25/kerneldense_25/bias*
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
GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_637689
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_24/kernel/Read/ReadVariableOp!dense_24/bias/Read/ReadVariableOp#dense_25/kernel/Read/ReadVariableOp!dense_25/bias/Read/ReadVariableOpConst*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_637943
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_24/kerneldense_24/biasdense_25/kerneldense_25/bias*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_637965??
?
?
.__inference_sequential_12_layer_call_fn_637647
input_13
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_6376362
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
input_13
?
?
.__inference_sequential_12_layer_call_fn_637829

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
GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_6376632
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
?
?
$__inference_signature_wrapper_637689
input_13
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8? **
f%R#
!__inference__wrapped_model_6375072
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
input_13
?
~
)__inference_dense_24_layer_call_fn_637869

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
GPU 2J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_6375422
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
?
?
.__inference_sequential_12_layer_call_fn_637674
input_13
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_13unknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_6376632
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
input_13
?E
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_637803

inputs.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource.
*dense_25_tensordot_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource
identity??
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_24/Tensordot/ReadVariableOp|
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_24/Tensordot/axes?
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_24/Tensordot/freej
dense_24/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_24/Tensordot/Shape?
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/GatherV2/axis?
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2?
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_24/Tensordot/GatherV2_1/axis?
dense_24/Tensordot/GatherV2_1GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/axes:output:0+dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2_1~
dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const?
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod?
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_1?
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod_1?
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_24/Tensordot/concat/axis?
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat?
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/stack?
dense_24/Tensordot/transpose	Transposeinputs"dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_24/Tensordot/transpose?
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_24/Tensordot/Reshape?
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/Tensordot/MatMul?
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_2?
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/concat_1/axis?
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1?
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_24/Tensordot?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_24/BiasAddw
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_24/Relu?
!dense_25/Tensordot/ReadVariableOpReadVariableOp*dense_25_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_25/Tensordot/ReadVariableOp|
dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/axes?
dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_25/Tensordot/free
dense_25/Tensordot/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:2
dense_25/Tensordot/Shape?
 dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/GatherV2/axis?
dense_25/Tensordot/GatherV2GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/free:output:0)dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2?
"dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_25/Tensordot/GatherV2_1/axis?
dense_25/Tensordot/GatherV2_1GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/axes:output:0+dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2_1~
dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const?
dense_25/Tensordot/ProdProd$dense_25/Tensordot/GatherV2:output:0!dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod?
dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const_1?
dense_25/Tensordot/Prod_1Prod&dense_25/Tensordot/GatherV2_1:output:0#dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod_1?
dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_25/Tensordot/concat/axis?
dense_25/Tensordot/concatConcatV2 dense_25/Tensordot/free:output:0 dense_25/Tensordot/axes:output:0'dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat?
dense_25/Tensordot/stackPack dense_25/Tensordot/Prod:output:0"dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/stack?
dense_25/Tensordot/transpose	Transposedense_24/Relu:activations:0"dense_25/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_25/Tensordot/transpose?
dense_25/Tensordot/ReshapeReshape dense_25/Tensordot/transpose:y:0!dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_25/Tensordot/Reshape?
dense_25/Tensordot/MatMulMatMul#dense_25/Tensordot/Reshape:output:0)dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/Tensordot/MatMul?
dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/Const_2?
 dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/concat_1/axis?
dense_25/Tensordot/concat_1ConcatV2$dense_25/Tensordot/GatherV2:output:0#dense_25/Tensordot/Const_2:output:0)dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat_1?
dense_25/TensordotReshape#dense_25/Tensordot/MatMul:product:0$dense_25/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_25/Tensordot?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/Tensordot:output:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_25/BiasAddq
IdentityIdentitydense_25/BiasAdd:output:0*
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
?
?
D__inference_dense_25_layer_call_and_return_conditional_losses_637588

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
?
?
D__inference_dense_25_layer_call_and_return_conditional_losses_637899

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
?
?
"__inference__traced_restore_637965
file_prefix$
 assignvariableop_dense_24_kernel$
 assignvariableop_1_dense_24_bias&
"assignvariableop_2_dense_25_kernel$
 assignvariableop_3_dense_25_bias

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
AssignVariableOpAssignVariableOp assignvariableop_dense_24_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_24_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_25_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_25_biasIdentity_3:output:0"/device:CPU:0*
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
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_637619
input_13
dense_24_637608
dense_24_637610
dense_25_637613
dense_25_637615
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_24_637608dense_24_637610*
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
GPU 2J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_6375422"
 dense_24/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_637613dense_25_637615*
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
GPU 2J 8? *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_6375882"
 dense_25/StatefulPartitionedCall?
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_13
?E
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_637746

inputs.
*dense_24_tensordot_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource.
*dense_25_tensordot_readvariableop_resource,
(dense_25_biasadd_readvariableop_resource
identity??
!dense_24/Tensordot/ReadVariableOpReadVariableOp*dense_24_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_24/Tensordot/ReadVariableOp|
dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_24/Tensordot/axes?
dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_24/Tensordot/freej
dense_24/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_24/Tensordot/Shape?
 dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/GatherV2/axis?
dense_24/Tensordot/GatherV2GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/free:output:0)dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2?
"dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_24/Tensordot/GatherV2_1/axis?
dense_24/Tensordot/GatherV2_1GatherV2!dense_24/Tensordot/Shape:output:0 dense_24/Tensordot/axes:output:0+dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_24/Tensordot/GatherV2_1~
dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const?
dense_24/Tensordot/ProdProd$dense_24/Tensordot/GatherV2:output:0!dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod?
dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_1?
dense_24/Tensordot/Prod_1Prod&dense_24/Tensordot/GatherV2_1:output:0#dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_24/Tensordot/Prod_1?
dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_24/Tensordot/concat/axis?
dense_24/Tensordot/concatConcatV2 dense_24/Tensordot/free:output:0 dense_24/Tensordot/axes:output:0'dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat?
dense_24/Tensordot/stackPack dense_24/Tensordot/Prod:output:0"dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/stack?
dense_24/Tensordot/transpose	Transposeinputs"dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_24/Tensordot/transpose?
dense_24/Tensordot/ReshapeReshape dense_24/Tensordot/transpose:y:0!dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_24/Tensordot/Reshape?
dense_24/Tensordot/MatMulMatMul#dense_24/Tensordot/Reshape:output:0)dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_24/Tensordot/MatMul?
dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_24/Tensordot/Const_2?
 dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_24/Tensordot/concat_1/axis?
dense_24/Tensordot/concat_1ConcatV2$dense_24/Tensordot/GatherV2:output:0#dense_24/Tensordot/Const_2:output:0)dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_24/Tensordot/concat_1?
dense_24/TensordotReshape#dense_24/Tensordot/MatMul:product:0$dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_24/Tensordot?
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_24/BiasAdd/ReadVariableOp?
dense_24/BiasAddBiasAdddense_24/Tensordot:output:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_24/BiasAddw
dense_24/ReluReludense_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_24/Relu?
!dense_25/Tensordot/ReadVariableOpReadVariableOp*dense_25_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_25/Tensordot/ReadVariableOp|
dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/axes?
dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_25/Tensordot/free
dense_25/Tensordot/ShapeShapedense_24/Relu:activations:0*
T0*
_output_shapes
:2
dense_25/Tensordot/Shape?
 dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/GatherV2/axis?
dense_25/Tensordot/GatherV2GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/free:output:0)dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2?
"dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_25/Tensordot/GatherV2_1/axis?
dense_25/Tensordot/GatherV2_1GatherV2!dense_25/Tensordot/Shape:output:0 dense_25/Tensordot/axes:output:0+dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_25/Tensordot/GatherV2_1~
dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const?
dense_25/Tensordot/ProdProd$dense_25/Tensordot/GatherV2:output:0!dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod?
dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_25/Tensordot/Const_1?
dense_25/Tensordot/Prod_1Prod&dense_25/Tensordot/GatherV2_1:output:0#dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_25/Tensordot/Prod_1?
dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_25/Tensordot/concat/axis?
dense_25/Tensordot/concatConcatV2 dense_25/Tensordot/free:output:0 dense_25/Tensordot/axes:output:0'dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat?
dense_25/Tensordot/stackPack dense_25/Tensordot/Prod:output:0"dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/stack?
dense_25/Tensordot/transpose	Transposedense_24/Relu:activations:0"dense_25/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_25/Tensordot/transpose?
dense_25/Tensordot/ReshapeReshape dense_25/Tensordot/transpose:y:0!dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_25/Tensordot/Reshape?
dense_25/Tensordot/MatMulMatMul#dense_25/Tensordot/Reshape:output:0)dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_25/Tensordot/MatMul?
dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_25/Tensordot/Const_2?
 dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_25/Tensordot/concat_1/axis?
dense_25/Tensordot/concat_1ConcatV2$dense_25/Tensordot/GatherV2:output:0#dense_25/Tensordot/Const_2:output:0)dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_25/Tensordot/concat_1?
dense_25/TensordotReshape#dense_25/Tensordot/MatMul:product:0$dense_25/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_25/Tensordot?
dense_25/BiasAdd/ReadVariableOpReadVariableOp(dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_25/BiasAdd/ReadVariableOp?
dense_25/BiasAddBiasAdddense_25/Tensordot:output:0'dense_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_25/BiasAddq
IdentityIdentitydense_25/BiasAdd:output:0*
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
?
~
)__inference_dense_25_layer_call_fn_637908

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
GPU 2J 8? *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_6375882
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
?W
?
!__inference__wrapped_model_637507
input_13<
8sequential_12_dense_24_tensordot_readvariableop_resource:
6sequential_12_dense_24_biasadd_readvariableop_resource<
8sequential_12_dense_25_tensordot_readvariableop_resource:
6sequential_12_dense_25_biasadd_readvariableop_resource
identity??
/sequential_12/dense_24/Tensordot/ReadVariableOpReadVariableOp8sequential_12_dense_24_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_12/dense_24/Tensordot/ReadVariableOp?
%sequential_12/dense_24/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_12/dense_24/Tensordot/axes?
%sequential_12/dense_24/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_12/dense_24/Tensordot/free?
&sequential_12/dense_24/Tensordot/ShapeShapeinput_13*
T0*
_output_shapes
:2(
&sequential_12/dense_24/Tensordot/Shape?
.sequential_12/dense_24/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_24/Tensordot/GatherV2/axis?
)sequential_12/dense_24/Tensordot/GatherV2GatherV2/sequential_12/dense_24/Tensordot/Shape:output:0.sequential_12/dense_24/Tensordot/free:output:07sequential_12/dense_24/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_12/dense_24/Tensordot/GatherV2?
0sequential_12/dense_24/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_12/dense_24/Tensordot/GatherV2_1/axis?
+sequential_12/dense_24/Tensordot/GatherV2_1GatherV2/sequential_12/dense_24/Tensordot/Shape:output:0.sequential_12/dense_24/Tensordot/axes:output:09sequential_12/dense_24/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_12/dense_24/Tensordot/GatherV2_1?
&sequential_12/dense_24/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_12/dense_24/Tensordot/Const?
%sequential_12/dense_24/Tensordot/ProdProd2sequential_12/dense_24/Tensordot/GatherV2:output:0/sequential_12/dense_24/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_12/dense_24/Tensordot/Prod?
(sequential_12/dense_24/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_12/dense_24/Tensordot/Const_1?
'sequential_12/dense_24/Tensordot/Prod_1Prod4sequential_12/dense_24/Tensordot/GatherV2_1:output:01sequential_12/dense_24/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_12/dense_24/Tensordot/Prod_1?
,sequential_12/dense_24/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_12/dense_24/Tensordot/concat/axis?
'sequential_12/dense_24/Tensordot/concatConcatV2.sequential_12/dense_24/Tensordot/free:output:0.sequential_12/dense_24/Tensordot/axes:output:05sequential_12/dense_24/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_12/dense_24/Tensordot/concat?
&sequential_12/dense_24/Tensordot/stackPack.sequential_12/dense_24/Tensordot/Prod:output:00sequential_12/dense_24/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_12/dense_24/Tensordot/stack?
*sequential_12/dense_24/Tensordot/transpose	Transposeinput_130sequential_12/dense_24/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22,
*sequential_12/dense_24/Tensordot/transpose?
(sequential_12/dense_24/Tensordot/ReshapeReshape.sequential_12/dense_24/Tensordot/transpose:y:0/sequential_12/dense_24/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_12/dense_24/Tensordot/Reshape?
'sequential_12/dense_24/Tensordot/MatMulMatMul1sequential_12/dense_24/Tensordot/Reshape:output:07sequential_12/dense_24/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_12/dense_24/Tensordot/MatMul?
(sequential_12/dense_24/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_12/dense_24/Tensordot/Const_2?
.sequential_12/dense_24/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_24/Tensordot/concat_1/axis?
)sequential_12/dense_24/Tensordot/concat_1ConcatV22sequential_12/dense_24/Tensordot/GatherV2:output:01sequential_12/dense_24/Tensordot/Const_2:output:07sequential_12/dense_24/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_12/dense_24/Tensordot/concat_1?
 sequential_12/dense_24/TensordotReshape1sequential_12/dense_24/Tensordot/MatMul:product:02sequential_12/dense_24/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2"
 sequential_12/dense_24/Tensordot?
-sequential_12/dense_24/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_12/dense_24/BiasAdd/ReadVariableOp?
sequential_12/dense_24/BiasAddBiasAdd)sequential_12/dense_24/Tensordot:output:05sequential_12/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2 
sequential_12/dense_24/BiasAdd?
sequential_12/dense_24/ReluRelu'sequential_12/dense_24/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
sequential_12/dense_24/Relu?
/sequential_12/dense_25/Tensordot/ReadVariableOpReadVariableOp8sequential_12_dense_25_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_12/dense_25/Tensordot/ReadVariableOp?
%sequential_12/dense_25/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_12/dense_25/Tensordot/axes?
%sequential_12/dense_25/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_12/dense_25/Tensordot/free?
&sequential_12/dense_25/Tensordot/ShapeShape)sequential_12/dense_24/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_12/dense_25/Tensordot/Shape?
.sequential_12/dense_25/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_25/Tensordot/GatherV2/axis?
)sequential_12/dense_25/Tensordot/GatherV2GatherV2/sequential_12/dense_25/Tensordot/Shape:output:0.sequential_12/dense_25/Tensordot/free:output:07sequential_12/dense_25/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_12/dense_25/Tensordot/GatherV2?
0sequential_12/dense_25/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_12/dense_25/Tensordot/GatherV2_1/axis?
+sequential_12/dense_25/Tensordot/GatherV2_1GatherV2/sequential_12/dense_25/Tensordot/Shape:output:0.sequential_12/dense_25/Tensordot/axes:output:09sequential_12/dense_25/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_12/dense_25/Tensordot/GatherV2_1?
&sequential_12/dense_25/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_12/dense_25/Tensordot/Const?
%sequential_12/dense_25/Tensordot/ProdProd2sequential_12/dense_25/Tensordot/GatherV2:output:0/sequential_12/dense_25/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_12/dense_25/Tensordot/Prod?
(sequential_12/dense_25/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_12/dense_25/Tensordot/Const_1?
'sequential_12/dense_25/Tensordot/Prod_1Prod4sequential_12/dense_25/Tensordot/GatherV2_1:output:01sequential_12/dense_25/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_12/dense_25/Tensordot/Prod_1?
,sequential_12/dense_25/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_12/dense_25/Tensordot/concat/axis?
'sequential_12/dense_25/Tensordot/concatConcatV2.sequential_12/dense_25/Tensordot/free:output:0.sequential_12/dense_25/Tensordot/axes:output:05sequential_12/dense_25/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_12/dense_25/Tensordot/concat?
&sequential_12/dense_25/Tensordot/stackPack.sequential_12/dense_25/Tensordot/Prod:output:00sequential_12/dense_25/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_12/dense_25/Tensordot/stack?
*sequential_12/dense_25/Tensordot/transpose	Transpose)sequential_12/dense_24/Relu:activations:00sequential_12/dense_25/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2,
*sequential_12/dense_25/Tensordot/transpose?
(sequential_12/dense_25/Tensordot/ReshapeReshape.sequential_12/dense_25/Tensordot/transpose:y:0/sequential_12/dense_25/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_12/dense_25/Tensordot/Reshape?
'sequential_12/dense_25/Tensordot/MatMulMatMul1sequential_12/dense_25/Tensordot/Reshape:output:07sequential_12/dense_25/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_12/dense_25/Tensordot/MatMul?
(sequential_12/dense_25/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_12/dense_25/Tensordot/Const_2?
.sequential_12/dense_25/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_12/dense_25/Tensordot/concat_1/axis?
)sequential_12/dense_25/Tensordot/concat_1ConcatV22sequential_12/dense_25/Tensordot/GatherV2:output:01sequential_12/dense_25/Tensordot/Const_2:output:07sequential_12/dense_25/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_12/dense_25/Tensordot/concat_1?
 sequential_12/dense_25/TensordotReshape1sequential_12/dense_25/Tensordot/MatMul:product:02sequential_12/dense_25/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22"
 sequential_12/dense_25/Tensordot?
-sequential_12/dense_25/BiasAdd/ReadVariableOpReadVariableOp6sequential_12_dense_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_12/dense_25/BiasAdd/ReadVariableOp?
sequential_12/dense_25/BiasAddBiasAdd)sequential_12/dense_25/Tensordot:output:05sequential_12/dense_25/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22 
sequential_12/dense_25/BiasAdd
IdentityIdentity'sequential_12/dense_25/BiasAdd:output:0*
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
input_13
?
?
.__inference_sequential_12_layer_call_fn_637816

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
GPU 2J 8? *R
fMRK
I__inference_sequential_12_layer_call_and_return_conditional_losses_6376362
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_637636

inputs
dense_24_637625
dense_24_637627
dense_25_637630
dense_25_637632
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinputsdense_24_637625dense_24_637627*
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
GPU 2J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_6375422"
 dense_24/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_637630dense_25_637632*
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
GPU 2J 8? *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_6375882"
 dense_25/StatefulPartitionedCall?
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
D__inference_dense_24_layer_call_and_return_conditional_losses_637860

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
?
?
__inference__traced_save_637943
file_prefix.
*savev2_dense_24_kernel_read_readvariableop,
(savev2_dense_24_bias_read_readvariableop.
*savev2_dense_25_kernel_read_readvariableop,
(savev2_dense_25_bias_read_readvariableop
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
value3B1 B+_temp_b51a96633ebb4e42b546beb1acd3fd24/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_24_kernel_read_readvariableop(savev2_dense_24_bias_read_readvariableop*savev2_dense_25_kernel_read_readvariableop(savev2_dense_25_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_637663

inputs
dense_24_637652
dense_24_637654
dense_25_637657
dense_25_637659
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinputsdense_24_637652dense_24_637654*
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
GPU 2J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_6375422"
 dense_24/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_637657dense_25_637659*
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
GPU 2J 8? *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_6375882"
 dense_25/StatefulPartitionedCall?
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
I__inference_sequential_12_layer_call_and_return_conditional_losses_637605
input_13
dense_24_637553
dense_24_637555
dense_25_637599
dense_25_637601
identity?? dense_24/StatefulPartitionedCall? dense_25/StatefulPartitionedCall?
 dense_24/StatefulPartitionedCallStatefulPartitionedCallinput_13dense_24_637553dense_24_637555*
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
GPU 2J 8? *M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_6375422"
 dense_24/StatefulPartitionedCall?
 dense_25/StatefulPartitionedCallStatefulPartitionedCall)dense_24/StatefulPartitionedCall:output:0dense_25_637599dense_25_637601*
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
GPU 2J 8? *M
fHRF
D__inference_dense_25_layer_call_and_return_conditional_losses_6375882"
 dense_25/StatefulPartitionedCall?
IdentityIdentity)dense_25/StatefulPartitionedCall:output:0!^dense_24/StatefulPartitionedCall!^dense_25/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2D
 dense_25/StatefulPartitionedCall dense_25/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_13
?
?
D__inference_dense_24_layer_call_and_return_conditional_losses_637542

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
input_135
serving_default_input_13:0?????????2@
dense_254
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_13"}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
)__call__
**&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_25", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 32]}}
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
!: 2dense_24/kernel
: 2dense_24/bias
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
!: 2dense_25/kernel
:2dense_25/bias
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
!__inference__wrapped_model_637507?
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
input_13?????????2
?2?
.__inference_sequential_12_layer_call_fn_637674
.__inference_sequential_12_layer_call_fn_637829
.__inference_sequential_12_layer_call_fn_637647
.__inference_sequential_12_layer_call_fn_637816?
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_637803
I__inference_sequential_12_layer_call_and_return_conditional_losses_637619
I__inference_sequential_12_layer_call_and_return_conditional_losses_637746
I__inference_sequential_12_layer_call_and_return_conditional_losses_637605?
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
)__inference_dense_24_layer_call_fn_637869?
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
D__inference_dense_24_layer_call_and_return_conditional_losses_637860?
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
)__inference_dense_25_layer_call_fn_637908?
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
D__inference_dense_25_layer_call_and_return_conditional_losses_637899?
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
4B2
$__inference_signature_wrapper_637689input_13?
!__inference__wrapped_model_637507v	
5?2
+?(
&?#
input_13?????????2
? "7?4
2
dense_25&?#
dense_25?????????2?
D__inference_dense_24_layer_call_and_return_conditional_losses_637860d	
3?0
)?&
$?!
inputs?????????2
? ")?&
?
0?????????2 
? ?
)__inference_dense_24_layer_call_fn_637869W	
3?0
)?&
$?!
inputs?????????2
? "??????????2 ?
D__inference_dense_25_layer_call_and_return_conditional_losses_637899d3?0
)?&
$?!
inputs?????????2 
? ")?&
?
0?????????2
? ?
)__inference_dense_25_layer_call_fn_637908W3?0
)?&
$?!
inputs?????????2 
? "??????????2?
I__inference_sequential_12_layer_call_and_return_conditional_losses_637605p	
=?:
3?0
&?#
input_13?????????2
p

 
? ")?&
?
0?????????2
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_637619p	
=?:
3?0
&?#
input_13?????????2
p 

 
? ")?&
?
0?????????2
? ?
I__inference_sequential_12_layer_call_and_return_conditional_losses_637746n	
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
I__inference_sequential_12_layer_call_and_return_conditional_losses_637803n	
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
.__inference_sequential_12_layer_call_fn_637647c	
=?:
3?0
&?#
input_13?????????2
p

 
? "??????????2?
.__inference_sequential_12_layer_call_fn_637674c	
=?:
3?0
&?#
input_13?????????2
p 

 
? "??????????2?
.__inference_sequential_12_layer_call_fn_637816a	
;?8
1?.
$?!
inputs?????????2
p

 
? "??????????2?
.__inference_sequential_12_layer_call_fn_637829a	
;?8
1?.
$?!
inputs?????????2
p 

 
? "??????????2?
$__inference_signature_wrapper_637689?	
A?>
? 
7?4
2
input_13&?#
input_13?????????2"7?4
2
dense_25&?#
dense_25?????????2