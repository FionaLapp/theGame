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
dense_32/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_32/kernel
s
#dense_32/kernel/Read/ReadVariableOpReadVariableOpdense_32/kernel*
_output_shapes

: *
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
: * 
shared_namedense_33/kernel
s
#dense_33/kernel/Read/ReadVariableOpReadVariableOpdense_33/kernel*
_output_shapes

: *
dtype0
r
dense_33/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_33/bias
k
!dense_33/bias/Read/ReadVariableOpReadVariableOpdense_33/bias*
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
VARIABLE_VALUEdense_32/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_32/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_33/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_33/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
serving_default_input_17Placeholder*+
_output_shapes
:?????????2*
dtype0* 
shape:?????????2
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_17dense_32/kerneldense_32/biasdense_33/kerneldense_33/bias*
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
%__inference_signature_wrapper_1874739
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpConst*
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
 __inference__traced_save_1874993
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_32/kerneldense_32/biasdense_33/kerneldense_33/bias*
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
#__inference__traced_restore_1875015??
?W
?
"__inference__wrapped_model_1874557
input_17<
8sequential_16_dense_32_tensordot_readvariableop_resource:
6sequential_16_dense_32_biasadd_readvariableop_resource<
8sequential_16_dense_33_tensordot_readvariableop_resource:
6sequential_16_dense_33_biasadd_readvariableop_resource
identity??
/sequential_16/dense_32/Tensordot/ReadVariableOpReadVariableOp8sequential_16_dense_32_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_16/dense_32/Tensordot/ReadVariableOp?
%sequential_16/dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_16/dense_32/Tensordot/axes?
%sequential_16/dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_16/dense_32/Tensordot/free?
&sequential_16/dense_32/Tensordot/ShapeShapeinput_17*
T0*
_output_shapes
:2(
&sequential_16/dense_32/Tensordot/Shape?
.sequential_16/dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_32/Tensordot/GatherV2/axis?
)sequential_16/dense_32/Tensordot/GatherV2GatherV2/sequential_16/dense_32/Tensordot/Shape:output:0.sequential_16/dense_32/Tensordot/free:output:07sequential_16/dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_16/dense_32/Tensordot/GatherV2?
0sequential_16/dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_16/dense_32/Tensordot/GatherV2_1/axis?
+sequential_16/dense_32/Tensordot/GatherV2_1GatherV2/sequential_16/dense_32/Tensordot/Shape:output:0.sequential_16/dense_32/Tensordot/axes:output:09sequential_16/dense_32/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_16/dense_32/Tensordot/GatherV2_1?
&sequential_16/dense_32/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_16/dense_32/Tensordot/Const?
%sequential_16/dense_32/Tensordot/ProdProd2sequential_16/dense_32/Tensordot/GatherV2:output:0/sequential_16/dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_16/dense_32/Tensordot/Prod?
(sequential_16/dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_32/Tensordot/Const_1?
'sequential_16/dense_32/Tensordot/Prod_1Prod4sequential_16/dense_32/Tensordot/GatherV2_1:output:01sequential_16/dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_16/dense_32/Tensordot/Prod_1?
,sequential_16/dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_16/dense_32/Tensordot/concat/axis?
'sequential_16/dense_32/Tensordot/concatConcatV2.sequential_16/dense_32/Tensordot/free:output:0.sequential_16/dense_32/Tensordot/axes:output:05sequential_16/dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_16/dense_32/Tensordot/concat?
&sequential_16/dense_32/Tensordot/stackPack.sequential_16/dense_32/Tensordot/Prod:output:00sequential_16/dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_16/dense_32/Tensordot/stack?
*sequential_16/dense_32/Tensordot/transpose	Transposeinput_170sequential_16/dense_32/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22,
*sequential_16/dense_32/Tensordot/transpose?
(sequential_16/dense_32/Tensordot/ReshapeReshape.sequential_16/dense_32/Tensordot/transpose:y:0/sequential_16/dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_16/dense_32/Tensordot/Reshape?
'sequential_16/dense_32/Tensordot/MatMulMatMul1sequential_16/dense_32/Tensordot/Reshape:output:07sequential_16/dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_16/dense_32/Tensordot/MatMul?
(sequential_16/dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_32/Tensordot/Const_2?
.sequential_16/dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_32/Tensordot/concat_1/axis?
)sequential_16/dense_32/Tensordot/concat_1ConcatV22sequential_16/dense_32/Tensordot/GatherV2:output:01sequential_16/dense_32/Tensordot/Const_2:output:07sequential_16/dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_16/dense_32/Tensordot/concat_1?
 sequential_16/dense_32/TensordotReshape1sequential_16/dense_32/Tensordot/MatMul:product:02sequential_16/dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2"
 sequential_16/dense_32/Tensordot?
-sequential_16/dense_32/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_16/dense_32/BiasAdd/ReadVariableOp?
sequential_16/dense_32/BiasAddBiasAdd)sequential_16/dense_32/Tensordot:output:05sequential_16/dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2 
sequential_16/dense_32/BiasAdd?
sequential_16/dense_32/ReluRelu'sequential_16/dense_32/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
sequential_16/dense_32/Relu?
/sequential_16/dense_33/Tensordot/ReadVariableOpReadVariableOp8sequential_16_dense_33_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_16/dense_33/Tensordot/ReadVariableOp?
%sequential_16/dense_33/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_16/dense_33/Tensordot/axes?
%sequential_16/dense_33/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_16/dense_33/Tensordot/free?
&sequential_16/dense_33/Tensordot/ShapeShape)sequential_16/dense_32/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_16/dense_33/Tensordot/Shape?
.sequential_16/dense_33/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_33/Tensordot/GatherV2/axis?
)sequential_16/dense_33/Tensordot/GatherV2GatherV2/sequential_16/dense_33/Tensordot/Shape:output:0.sequential_16/dense_33/Tensordot/free:output:07sequential_16/dense_33/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_16/dense_33/Tensordot/GatherV2?
0sequential_16/dense_33/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_16/dense_33/Tensordot/GatherV2_1/axis?
+sequential_16/dense_33/Tensordot/GatherV2_1GatherV2/sequential_16/dense_33/Tensordot/Shape:output:0.sequential_16/dense_33/Tensordot/axes:output:09sequential_16/dense_33/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_16/dense_33/Tensordot/GatherV2_1?
&sequential_16/dense_33/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_16/dense_33/Tensordot/Const?
%sequential_16/dense_33/Tensordot/ProdProd2sequential_16/dense_33/Tensordot/GatherV2:output:0/sequential_16/dense_33/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_16/dense_33/Tensordot/Prod?
(sequential_16/dense_33/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_16/dense_33/Tensordot/Const_1?
'sequential_16/dense_33/Tensordot/Prod_1Prod4sequential_16/dense_33/Tensordot/GatherV2_1:output:01sequential_16/dense_33/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_16/dense_33/Tensordot/Prod_1?
,sequential_16/dense_33/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_16/dense_33/Tensordot/concat/axis?
'sequential_16/dense_33/Tensordot/concatConcatV2.sequential_16/dense_33/Tensordot/free:output:0.sequential_16/dense_33/Tensordot/axes:output:05sequential_16/dense_33/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_16/dense_33/Tensordot/concat?
&sequential_16/dense_33/Tensordot/stackPack.sequential_16/dense_33/Tensordot/Prod:output:00sequential_16/dense_33/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_16/dense_33/Tensordot/stack?
*sequential_16/dense_33/Tensordot/transpose	Transpose)sequential_16/dense_32/Relu:activations:00sequential_16/dense_33/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2,
*sequential_16/dense_33/Tensordot/transpose?
(sequential_16/dense_33/Tensordot/ReshapeReshape.sequential_16/dense_33/Tensordot/transpose:y:0/sequential_16/dense_33/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_16/dense_33/Tensordot/Reshape?
'sequential_16/dense_33/Tensordot/MatMulMatMul1sequential_16/dense_33/Tensordot/Reshape:output:07sequential_16/dense_33/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_16/dense_33/Tensordot/MatMul?
(sequential_16/dense_33/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_16/dense_33/Tensordot/Const_2?
.sequential_16/dense_33/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_16/dense_33/Tensordot/concat_1/axis?
)sequential_16/dense_33/Tensordot/concat_1ConcatV22sequential_16/dense_33/Tensordot/GatherV2:output:01sequential_16/dense_33/Tensordot/Const_2:output:07sequential_16/dense_33/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_16/dense_33/Tensordot/concat_1?
 sequential_16/dense_33/TensordotReshape1sequential_16/dense_33/Tensordot/MatMul:product:02sequential_16/dense_33/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22"
 sequential_16/dense_33/Tensordot?
-sequential_16/dense_33/BiasAdd/ReadVariableOpReadVariableOp6sequential_16_dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_16/dense_33/BiasAdd/ReadVariableOp?
sequential_16/dense_33/BiasAddBiasAdd)sequential_16/dense_33/Tensordot:output:05sequential_16/dense_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22 
sequential_16/dense_33/BiasAdd
IdentityIdentity'sequential_16/dense_33/BiasAdd:output:0*
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
input_17
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874713

inputs
dense_32_1874702
dense_32_1874704
dense_33_1874707
dense_33_1874709
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_1874702dense_32_1874704*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_18745922"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_1874707dense_33_1874709*
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
E__inference_dense_33_layer_call_and_return_conditional_losses_18746382"
 dense_33/StatefulPartitionedCall?
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
E__inference_dense_33_layer_call_and_return_conditional_losses_1874638

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
?

*__inference_dense_32_layer_call_fn_1874919

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
E__inference_dense_32_layer_call_and_return_conditional_losses_18745922
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
?E
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874853

inputs.
*dense_32_tensordot_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource.
*dense_33_tensordot_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource
identity??
!dense_32/Tensordot/ReadVariableOpReadVariableOp*dense_32_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_32/Tensordot/ReadVariableOp|
dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/axes?
dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_32/Tensordot/freej
dense_32/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_32/Tensordot/Shape?
 dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/GatherV2/axis?
dense_32/Tensordot/GatherV2GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/free:output:0)dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2?
"dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_32/Tensordot/GatherV2_1/axis?
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
dense_32/Tensordot/Const?
dense_32/Tensordot/ProdProd$dense_32/Tensordot/GatherV2:output:0!dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod?
dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_1?
dense_32/Tensordot/Prod_1Prod&dense_32/Tensordot/GatherV2_1:output:0#dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod_1?
dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_32/Tensordot/concat/axis?
dense_32/Tensordot/concatConcatV2 dense_32/Tensordot/free:output:0 dense_32/Tensordot/axes:output:0'dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat?
dense_32/Tensordot/stackPack dense_32/Tensordot/Prod:output:0"dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/stack?
dense_32/Tensordot/transpose	Transposeinputs"dense_32/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_32/Tensordot/transpose?
dense_32/Tensordot/ReshapeReshape dense_32/Tensordot/transpose:y:0!dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_32/Tensordot/Reshape?
dense_32/Tensordot/MatMulMatMul#dense_32/Tensordot/Reshape:output:0)dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_32/Tensordot/MatMul?
dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_2?
 dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/concat_1/axis?
dense_32/Tensordot/concat_1ConcatV2$dense_32/Tensordot/GatherV2:output:0#dense_32/Tensordot/Const_2:output:0)dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat_1?
dense_32/TensordotReshape#dense_32/Tensordot/MatMul:product:0$dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_32/Tensordot?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/Tensordot:output:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_32/BiasAddw
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_32/Relu?
!dense_33/Tensordot/ReadVariableOpReadVariableOp*dense_33_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_33/Tensordot/ReadVariableOp|
dense_33/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_33/Tensordot/axes?
dense_33/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_33/Tensordot/free
dense_33/Tensordot/ShapeShapedense_32/Relu:activations:0*
T0*
_output_shapes
:2
dense_33/Tensordot/Shape?
 dense_33/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_33/Tensordot/GatherV2/axis?
dense_33/Tensordot/GatherV2GatherV2!dense_33/Tensordot/Shape:output:0 dense_33/Tensordot/free:output:0)dense_33/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_33/Tensordot/GatherV2?
"dense_33/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_33/Tensordot/GatherV2_1/axis?
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
dense_33/Tensordot/Const?
dense_33/Tensordot/ProdProd$dense_33/Tensordot/GatherV2:output:0!dense_33/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_33/Tensordot/Prod?
dense_33/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_33/Tensordot/Const_1?
dense_33/Tensordot/Prod_1Prod&dense_33/Tensordot/GatherV2_1:output:0#dense_33/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_33/Tensordot/Prod_1?
dense_33/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_33/Tensordot/concat/axis?
dense_33/Tensordot/concatConcatV2 dense_33/Tensordot/free:output:0 dense_33/Tensordot/axes:output:0'dense_33/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/concat?
dense_33/Tensordot/stackPack dense_33/Tensordot/Prod:output:0"dense_33/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/stack?
dense_33/Tensordot/transpose	Transposedense_32/Relu:activations:0"dense_33/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_33/Tensordot/transpose?
dense_33/Tensordot/ReshapeReshape dense_33/Tensordot/transpose:y:0!dense_33/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_33/Tensordot/Reshape?
dense_33/Tensordot/MatMulMatMul#dense_33/Tensordot/Reshape:output:0)dense_33/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_33/Tensordot/MatMul?
dense_33/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_33/Tensordot/Const_2?
 dense_33/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_33/Tensordot/concat_1/axis?
dense_33/Tensordot/concat_1ConcatV2$dense_33/Tensordot/GatherV2:output:0#dense_33/Tensordot/Const_2:output:0)dense_33/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/concat_1?
dense_33/TensordotReshape#dense_33/Tensordot/MatMul:product:0$dense_33/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_33/Tensordot?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/Tensordot:output:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_33/BiasAddq
IdentityIdentitydense_33/BiasAdd:output:0*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_1874592

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
 __inference__traced_save_1874993
file_prefix.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop
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
value3B1 B+_temp_7a95e2cc30b34b7cb91e88214f6ddc20/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?E
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874796

inputs.
*dense_32_tensordot_readvariableop_resource,
(dense_32_biasadd_readvariableop_resource.
*dense_33_tensordot_readvariableop_resource,
(dense_33_biasadd_readvariableop_resource
identity??
!dense_32/Tensordot/ReadVariableOpReadVariableOp*dense_32_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_32/Tensordot/ReadVariableOp|
dense_32/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_32/Tensordot/axes?
dense_32/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_32/Tensordot/freej
dense_32/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_32/Tensordot/Shape?
 dense_32/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/GatherV2/axis?
dense_32/Tensordot/GatherV2GatherV2!dense_32/Tensordot/Shape:output:0 dense_32/Tensordot/free:output:0)dense_32/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_32/Tensordot/GatherV2?
"dense_32/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_32/Tensordot/GatherV2_1/axis?
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
dense_32/Tensordot/Const?
dense_32/Tensordot/ProdProd$dense_32/Tensordot/GatherV2:output:0!dense_32/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod?
dense_32/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_1?
dense_32/Tensordot/Prod_1Prod&dense_32/Tensordot/GatherV2_1:output:0#dense_32/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_32/Tensordot/Prod_1?
dense_32/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_32/Tensordot/concat/axis?
dense_32/Tensordot/concatConcatV2 dense_32/Tensordot/free:output:0 dense_32/Tensordot/axes:output:0'dense_32/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat?
dense_32/Tensordot/stackPack dense_32/Tensordot/Prod:output:0"dense_32/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/stack?
dense_32/Tensordot/transpose	Transposeinputs"dense_32/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_32/Tensordot/transpose?
dense_32/Tensordot/ReshapeReshape dense_32/Tensordot/transpose:y:0!dense_32/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_32/Tensordot/Reshape?
dense_32/Tensordot/MatMulMatMul#dense_32/Tensordot/Reshape:output:0)dense_32/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_32/Tensordot/MatMul?
dense_32/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_32/Tensordot/Const_2?
 dense_32/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_32/Tensordot/concat_1/axis?
dense_32/Tensordot/concat_1ConcatV2$dense_32/Tensordot/GatherV2:output:0#dense_32/Tensordot/Const_2:output:0)dense_32/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_32/Tensordot/concat_1?
dense_32/TensordotReshape#dense_32/Tensordot/MatMul:product:0$dense_32/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_32/Tensordot?
dense_32/BiasAdd/ReadVariableOpReadVariableOp(dense_32_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_32/BiasAdd/ReadVariableOp?
dense_32/BiasAddBiasAdddense_32/Tensordot:output:0'dense_32/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_32/BiasAddw
dense_32/ReluReludense_32/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_32/Relu?
!dense_33/Tensordot/ReadVariableOpReadVariableOp*dense_33_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_33/Tensordot/ReadVariableOp|
dense_33/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_33/Tensordot/axes?
dense_33/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_33/Tensordot/free
dense_33/Tensordot/ShapeShapedense_32/Relu:activations:0*
T0*
_output_shapes
:2
dense_33/Tensordot/Shape?
 dense_33/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_33/Tensordot/GatherV2/axis?
dense_33/Tensordot/GatherV2GatherV2!dense_33/Tensordot/Shape:output:0 dense_33/Tensordot/free:output:0)dense_33/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_33/Tensordot/GatherV2?
"dense_33/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_33/Tensordot/GatherV2_1/axis?
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
dense_33/Tensordot/Const?
dense_33/Tensordot/ProdProd$dense_33/Tensordot/GatherV2:output:0!dense_33/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_33/Tensordot/Prod?
dense_33/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_33/Tensordot/Const_1?
dense_33/Tensordot/Prod_1Prod&dense_33/Tensordot/GatherV2_1:output:0#dense_33/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_33/Tensordot/Prod_1?
dense_33/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_33/Tensordot/concat/axis?
dense_33/Tensordot/concatConcatV2 dense_33/Tensordot/free:output:0 dense_33/Tensordot/axes:output:0'dense_33/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/concat?
dense_33/Tensordot/stackPack dense_33/Tensordot/Prod:output:0"dense_33/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/stack?
dense_33/Tensordot/transpose	Transposedense_32/Relu:activations:0"dense_33/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_33/Tensordot/transpose?
dense_33/Tensordot/ReshapeReshape dense_33/Tensordot/transpose:y:0!dense_33/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_33/Tensordot/Reshape?
dense_33/Tensordot/MatMulMatMul#dense_33/Tensordot/Reshape:output:0)dense_33/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_33/Tensordot/MatMul?
dense_33/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_33/Tensordot/Const_2?
 dense_33/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_33/Tensordot/concat_1/axis?
dense_33/Tensordot/concat_1ConcatV2$dense_33/Tensordot/GatherV2:output:0#dense_33/Tensordot/Const_2:output:0)dense_33/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_33/Tensordot/concat_1?
dense_33/TensordotReshape#dense_33/Tensordot/MatMul:product:0$dense_33/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_33/Tensordot?
dense_33/BiasAdd/ReadVariableOpReadVariableOp(dense_33_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_33/BiasAdd/ReadVariableOp?
dense_33/BiasAddBiasAdddense_33/Tensordot:output:0'dense_33/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_33/BiasAddq
IdentityIdentitydense_33/BiasAdd:output:0*
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
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874686

inputs
dense_32_1874675
dense_32_1874677
dense_33_1874680
dense_33_1874682
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_1874675dense_32_1874677*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_18745922"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_1874680dense_33_1874682*
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
E__inference_dense_33_layer_call_and_return_conditional_losses_18746382"
 dense_33/StatefulPartitionedCall?
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
/__inference_sequential_16_layer_call_fn_1874724
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2*
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_18747132
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
input_17
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874655
input_17
dense_32_1874603
dense_32_1874605
dense_33_1874649
dense_33_1874651
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinput_17dense_32_1874603dense_32_1874605*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_18745922"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_1874649dense_33_1874651*
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
E__inference_dense_33_layer_call_and_return_conditional_losses_18746382"
 dense_33/StatefulPartitionedCall?
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_17
?
?
%__inference_signature_wrapper_1874739
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2*
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
"__inference__wrapped_model_18745572
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
input_17
?
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874669
input_17
dense_32_1874658
dense_32_1874660
dense_33_1874663
dense_33_1874665
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinput_17dense_32_1874658dense_32_1874660*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_18745922"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_1874663dense_33_1874665*
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
E__inference_dense_33_layer_call_and_return_conditional_losses_18746382"
 dense_33/StatefulPartitionedCall?
IdentityIdentity)dense_33/StatefulPartitionedCall:output:0!^dense_32/StatefulPartitionedCall!^dense_33/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_32/StatefulPartitionedCall dense_32/StatefulPartitionedCall2D
 dense_33/StatefulPartitionedCall dense_33/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_17
?

*__inference_dense_33_layer_call_fn_1874958

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
E__inference_dense_33_layer_call_and_return_conditional_losses_18746382
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
?
?
/__inference_sequential_16_layer_call_fn_1874697
input_17
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_17unknown	unknown_0	unknown_1	unknown_2*
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_18746862
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
input_17
?
?
#__inference__traced_restore_1875015
file_prefix$
 assignvariableop_dense_32_kernel$
 assignvariableop_1_dense_32_bias&
"assignvariableop_2_dense_33_kernel$
 assignvariableop_3_dense_33_bias

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
AssignVariableOpAssignVariableOp assignvariableop_dense_32_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_32_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_33_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_33_biasIdentity_3:output:0"/device:CPU:0*
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
?
E__inference_dense_33_layer_call_and_return_conditional_losses_1874949

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
E__inference_dense_32_layer_call_and_return_conditional_losses_1874910

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
?
?
/__inference_sequential_16_layer_call_fn_1874879

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
J__inference_sequential_16_layer_call_and_return_conditional_losses_18747132
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
/__inference_sequential_16_layer_call_fn_1874866

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
J__inference_sequential_16_layer_call_and_return_conditional_losses_18746862
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
input_175
serving_default_input_17:0?????????2@
dense_334
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.001, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
)__call__
**&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 32]}}
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
!: 2dense_32/kernel
: 2dense_32/bias
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
!: 2dense_33/kernel
:2dense_33/bias
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
"__inference__wrapped_model_1874557?
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
input_17?????????2
?2?
/__inference_sequential_16_layer_call_fn_1874879
/__inference_sequential_16_layer_call_fn_1874697
/__inference_sequential_16_layer_call_fn_1874724
/__inference_sequential_16_layer_call_fn_1874866?
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874796
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874655
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874669
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874853?
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
*__inference_dense_32_layer_call_fn_1874919?
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
E__inference_dense_32_layer_call_and_return_conditional_losses_1874910?
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
*__inference_dense_33_layer_call_fn_1874958?
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
E__inference_dense_33_layer_call_and_return_conditional_losses_1874949?
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
%__inference_signature_wrapper_1874739input_17?
"__inference__wrapped_model_1874557v	
5?2
+?(
&?#
input_17?????????2
? "7?4
2
dense_33&?#
dense_33?????????2?
E__inference_dense_32_layer_call_and_return_conditional_losses_1874910d	
3?0
)?&
$?!
inputs?????????2
? ")?&
?
0?????????2 
? ?
*__inference_dense_32_layer_call_fn_1874919W	
3?0
)?&
$?!
inputs?????????2
? "??????????2 ?
E__inference_dense_33_layer_call_and_return_conditional_losses_1874949d3?0
)?&
$?!
inputs?????????2 
? ")?&
?
0?????????2
? ?
*__inference_dense_33_layer_call_fn_1874958W3?0
)?&
$?!
inputs?????????2 
? "??????????2?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874655p	
=?:
3?0
&?#
input_17?????????2
p

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874669p	
=?:
3?0
&?#
input_17?????????2
p 

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874796n	
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_1874853n	
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
/__inference_sequential_16_layer_call_fn_1874697c	
=?:
3?0
&?#
input_17?????????2
p

 
? "??????????2?
/__inference_sequential_16_layer_call_fn_1874724c	
=?:
3?0
&?#
input_17?????????2
p 

 
? "??????????2?
/__inference_sequential_16_layer_call_fn_1874866a	
;?8
1?.
$?!
inputs?????????2
p

 
? "??????????2?
/__inference_sequential_16_layer_call_fn_1874879a	
;?8
1?.
$?!
inputs?????????2
p 

 
? "??????????2?
%__inference_signature_wrapper_1874739?	
A?>
? 
7?4
2
input_17&?#
input_17?????????2"7?4
2
dense_33&?#
dense_33?????????2