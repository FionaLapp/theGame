??
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
 ?"serve*2.3.02unknown8??
z
dense_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_28/kernel
s
#dense_28/kernel/Read/ReadVariableOpReadVariableOpdense_28/kernel*
_output_shapes

: *
dtype0
r
dense_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_28/bias
k
!dense_28/bias/Read/ReadVariableOpReadVariableOpdense_28/bias*
_output_shapes
: *
dtype0
z
dense_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_29/kernel
s
#dense_29/kernel/Read/ReadVariableOpReadVariableOpdense_29/kernel*
_output_shapes

: *
dtype0
r
dense_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_29/bias
k
!dense_29/bias/Read/ReadVariableOpReadVariableOpdense_29/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_28/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_28/kernel/m
?
*Adam/dense_28/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_28/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_28/bias/m
y
(Adam/dense_28/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_29/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_29/kernel/m
?
*Adam/dense_29/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_29/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/m
y
(Adam/dense_29/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_28/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_28/kernel/v
?
*Adam/dense_28/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_28/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_28/bias/v
y
(Adam/dense_28/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_28/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_29/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_29/kernel/v
?
*Adam/dense_29/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_29/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_29/bias/v
y
(Adam/dense_29/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_29/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
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
?
iter

beta_1

beta_2
	decay
learning_rate	m4
m5m6m7	v8
v9v:v;

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
non_trainable_variables
	variables

layers
regularization_losses
metrics
layer_metrics
layer_regularization_losses
 
[Y
VARIABLE_VALUEdense_28/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_28/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
non_trainable_variables
	variables

 layers
regularization_losses
!metrics
"layer_metrics
#layer_regularization_losses
[Y
VARIABLE_VALUEdense_29/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_29/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
trainable_variables
$non_trainable_variables
	variables

%layers
regularization_losses
&metrics
'layer_metrics
(layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

)0
*1
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
4
	+total
	,count
-	variables
.	keras_api
D
	/total
	0count
1
_fn_kwargs
2	variables
3	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

-	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

2	variables
~|
VARIABLE_VALUEAdam/dense_28/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_28/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_28/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_29/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_29/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_15Placeholder*+
_output_shapes
:?????????2*
dtype0* 
shape:?????????2
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_15dense_28/kerneldense_28/biasdense_29/kerneldense_29/bias*
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
%__inference_signature_wrapper_1873775
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_28/kernel/Read/ReadVariableOp!dense_28/bias/Read/ReadVariableOp#dense_29/kernel/Read/ReadVariableOp!dense_29/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_28/kernel/m/Read/ReadVariableOp(Adam/dense_28/bias/m/Read/ReadVariableOp*Adam/dense_29/kernel/m/Read/ReadVariableOp(Adam/dense_29/bias/m/Read/ReadVariableOp*Adam/dense_28/kernel/v/Read/ReadVariableOp(Adam/dense_28/bias/v/Read/ReadVariableOp*Adam/dense_29/kernel/v/Read/ReadVariableOp(Adam/dense_29/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
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
 __inference__traced_save_1874080
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_28/kerneldense_28/biasdense_29/kerneldense_29/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_28/kernel/mAdam/dense_28/bias/mAdam/dense_29/kernel/mAdam/dense_29/bias/mAdam/dense_28/kernel/vAdam/dense_28/bias/vAdam/dense_29/kernel/vAdam/dense_29/bias/v*!
Tin
2*
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
#__inference__traced_restore_1874153??
?
?
E__inference_dense_29_layer_call_and_return_conditional_losses_1873666

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
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873832

inputs.
*dense_28_tensordot_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource.
*dense_29_tensordot_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource
identity??
!dense_28/Tensordot/ReadVariableOpReadVariableOp*dense_28_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_28/Tensordot/ReadVariableOp|
dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_28/Tensordot/axes?
dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_28/Tensordot/freej
dense_28/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_28/Tensordot/Shape?
 dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/GatherV2/axis?
dense_28/Tensordot/GatherV2GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/free:output:0)dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2?
"dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_28/Tensordot/GatherV2_1/axis?
dense_28/Tensordot/GatherV2_1GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/axes:output:0+dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2_1~
dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const?
dense_28/Tensordot/ProdProd$dense_28/Tensordot/GatherV2:output:0!dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod?
dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const_1?
dense_28/Tensordot/Prod_1Prod&dense_28/Tensordot/GatherV2_1:output:0#dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod_1?
dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_28/Tensordot/concat/axis?
dense_28/Tensordot/concatConcatV2 dense_28/Tensordot/free:output:0 dense_28/Tensordot/axes:output:0'dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat?
dense_28/Tensordot/stackPack dense_28/Tensordot/Prod:output:0"dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/stack?
dense_28/Tensordot/transpose	Transposeinputs"dense_28/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_28/Tensordot/transpose?
dense_28/Tensordot/ReshapeReshape dense_28/Tensordot/transpose:y:0!dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_28/Tensordot/Reshape?
dense_28/Tensordot/MatMulMatMul#dense_28/Tensordot/Reshape:output:0)dense_28/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_28/Tensordot/MatMul?
dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const_2?
 dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/concat_1/axis?
dense_28/Tensordot/concat_1ConcatV2$dense_28/Tensordot/GatherV2:output:0#dense_28/Tensordot/Const_2:output:0)dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat_1?
dense_28/TensordotReshape#dense_28/Tensordot/MatMul:product:0$dense_28/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_28/Tensordot?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/Tensordot:output:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_28/BiasAddw
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_28/Relu?
!dense_29/Tensordot/ReadVariableOpReadVariableOp*dense_29_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_29/Tensordot/ReadVariableOp|
dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_29/Tensordot/axes?
dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_29/Tensordot/free
dense_29/Tensordot/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:2
dense_29/Tensordot/Shape?
 dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/GatherV2/axis?
dense_29/Tensordot/GatherV2GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/free:output:0)dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2?
"dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_29/Tensordot/GatherV2_1/axis?
dense_29/Tensordot/GatherV2_1GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/axes:output:0+dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2_1~
dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const?
dense_29/Tensordot/ProdProd$dense_29/Tensordot/GatherV2:output:0!dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod?
dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const_1?
dense_29/Tensordot/Prod_1Prod&dense_29/Tensordot/GatherV2_1:output:0#dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod_1?
dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_29/Tensordot/concat/axis?
dense_29/Tensordot/concatConcatV2 dense_29/Tensordot/free:output:0 dense_29/Tensordot/axes:output:0'dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat?
dense_29/Tensordot/stackPack dense_29/Tensordot/Prod:output:0"dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/stack?
dense_29/Tensordot/transpose	Transposedense_28/Relu:activations:0"dense_29/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_29/Tensordot/transpose?
dense_29/Tensordot/ReshapeReshape dense_29/Tensordot/transpose:y:0!dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_29/Tensordot/Reshape?
dense_29/Tensordot/MatMulMatMul#dense_29/Tensordot/Reshape:output:0)dense_29/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_29/Tensordot/MatMul?
dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_29/Tensordot/Const_2?
 dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/concat_1/axis?
dense_29/Tensordot/concat_1ConcatV2$dense_29/Tensordot/GatherV2:output:0#dense_29/Tensordot/Const_2:output:0)dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat_1?
dense_29/TensordotReshape#dense_29/Tensordot/MatMul:product:0$dense_29/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_29/Tensordot?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/Tensordot:output:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_29/BiasAddq
IdentityIdentitydense_29/BiasAdd:output:0*
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
?E
?
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873889

inputs.
*dense_28_tensordot_readvariableop_resource,
(dense_28_biasadd_readvariableop_resource.
*dense_29_tensordot_readvariableop_resource,
(dense_29_biasadd_readvariableop_resource
identity??
!dense_28/Tensordot/ReadVariableOpReadVariableOp*dense_28_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_28/Tensordot/ReadVariableOp|
dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_28/Tensordot/axes?
dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_28/Tensordot/freej
dense_28/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_28/Tensordot/Shape?
 dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/GatherV2/axis?
dense_28/Tensordot/GatherV2GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/free:output:0)dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2?
"dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_28/Tensordot/GatherV2_1/axis?
dense_28/Tensordot/GatherV2_1GatherV2!dense_28/Tensordot/Shape:output:0 dense_28/Tensordot/axes:output:0+dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_28/Tensordot/GatherV2_1~
dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const?
dense_28/Tensordot/ProdProd$dense_28/Tensordot/GatherV2:output:0!dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod?
dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const_1?
dense_28/Tensordot/Prod_1Prod&dense_28/Tensordot/GatherV2_1:output:0#dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_28/Tensordot/Prod_1?
dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_28/Tensordot/concat/axis?
dense_28/Tensordot/concatConcatV2 dense_28/Tensordot/free:output:0 dense_28/Tensordot/axes:output:0'dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat?
dense_28/Tensordot/stackPack dense_28/Tensordot/Prod:output:0"dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/stack?
dense_28/Tensordot/transpose	Transposeinputs"dense_28/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_28/Tensordot/transpose?
dense_28/Tensordot/ReshapeReshape dense_28/Tensordot/transpose:y:0!dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_28/Tensordot/Reshape?
dense_28/Tensordot/MatMulMatMul#dense_28/Tensordot/Reshape:output:0)dense_28/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_28/Tensordot/MatMul?
dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_28/Tensordot/Const_2?
 dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_28/Tensordot/concat_1/axis?
dense_28/Tensordot/concat_1ConcatV2$dense_28/Tensordot/GatherV2:output:0#dense_28/Tensordot/Const_2:output:0)dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_28/Tensordot/concat_1?
dense_28/TensordotReshape#dense_28/Tensordot/MatMul:product:0$dense_28/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_28/Tensordot?
dense_28/BiasAdd/ReadVariableOpReadVariableOp(dense_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_28/BiasAdd/ReadVariableOp?
dense_28/BiasAddBiasAdddense_28/Tensordot:output:0'dense_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_28/BiasAddw
dense_28/ReluReludense_28/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_28/Relu?
!dense_29/Tensordot/ReadVariableOpReadVariableOp*dense_29_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_29/Tensordot/ReadVariableOp|
dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_29/Tensordot/axes?
dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_29/Tensordot/free
dense_29/Tensordot/ShapeShapedense_28/Relu:activations:0*
T0*
_output_shapes
:2
dense_29/Tensordot/Shape?
 dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/GatherV2/axis?
dense_29/Tensordot/GatherV2GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/free:output:0)dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2?
"dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_29/Tensordot/GatherV2_1/axis?
dense_29/Tensordot/GatherV2_1GatherV2!dense_29/Tensordot/Shape:output:0 dense_29/Tensordot/axes:output:0+dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_29/Tensordot/GatherV2_1~
dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const?
dense_29/Tensordot/ProdProd$dense_29/Tensordot/GatherV2:output:0!dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod?
dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_29/Tensordot/Const_1?
dense_29/Tensordot/Prod_1Prod&dense_29/Tensordot/GatherV2_1:output:0#dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_29/Tensordot/Prod_1?
dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_29/Tensordot/concat/axis?
dense_29/Tensordot/concatConcatV2 dense_29/Tensordot/free:output:0 dense_29/Tensordot/axes:output:0'dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat?
dense_29/Tensordot/stackPack dense_29/Tensordot/Prod:output:0"dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/stack?
dense_29/Tensordot/transpose	Transposedense_28/Relu:activations:0"dense_29/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_29/Tensordot/transpose?
dense_29/Tensordot/ReshapeReshape dense_29/Tensordot/transpose:y:0!dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_29/Tensordot/Reshape?
dense_29/Tensordot/MatMulMatMul#dense_29/Tensordot/Reshape:output:0)dense_29/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_29/Tensordot/MatMul?
dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_29/Tensordot/Const_2?
 dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_29/Tensordot/concat_1/axis?
dense_29/Tensordot/concat_1ConcatV2$dense_29/Tensordot/GatherV2:output:0#dense_29/Tensordot/Const_2:output:0)dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_29/Tensordot/concat_1?
dense_29/TensordotReshape#dense_29/Tensordot/MatMul:product:0$dense_29/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_29/Tensordot?
dense_29/BiasAdd/ReadVariableOpReadVariableOp(dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_29/BiasAdd/ReadVariableOp?
dense_29/BiasAddBiasAdddense_29/Tensordot:output:0'dense_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_29/BiasAddq
IdentityIdentitydense_29/BiasAdd:output:0*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_1873620

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
*__inference_dense_28_layer_call_fn_1873955

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
E__inference_dense_28_layer_call_and_return_conditional_losses_18736202
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
%__inference_signature_wrapper_1873775
input_15
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2*
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
"__inference__wrapped_model_18735852
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
input_15
?3
?
 __inference__traced_save_1874080
file_prefix.
*savev2_dense_28_kernel_read_readvariableop,
(savev2_dense_28_bias_read_readvariableop.
*savev2_dense_29_kernel_read_readvariableop,
(savev2_dense_29_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_28_kernel_m_read_readvariableop3
/savev2_adam_dense_28_bias_m_read_readvariableop5
1savev2_adam_dense_29_kernel_m_read_readvariableop3
/savev2_adam_dense_29_bias_m_read_readvariableop5
1savev2_adam_dense_28_kernel_v_read_readvariableop3
/savev2_adam_dense_28_bias_v_read_readvariableop5
1savev2_adam_dense_29_kernel_v_read_readvariableop3
/savev2_adam_dense_29_bias_v_read_readvariableop
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
value3B1 B+_temp_ab2e81ef36b2477ca206fa20f24e4734/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_28_kernel_read_readvariableop(savev2_dense_28_bias_read_readvariableop*savev2_dense_29_kernel_read_readvariableop(savev2_dense_29_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_28_kernel_m_read_readvariableop/savev2_adam_dense_28_bias_m_read_readvariableop1savev2_adam_dense_29_kernel_m_read_readvariableop/savev2_adam_dense_29_bias_m_read_readvariableop1savev2_adam_dense_28_kernel_v_read_readvariableop/savev2_adam_dense_28_bias_v_read_readvariableop1savev2_adam_dense_29_kernel_v_read_readvariableop/savev2_adam_dense_29_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
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

identity_1Identity_1:output:0*?
_input_shapesx
v: : : : :: : : : : : : : : : : : :: : : :: 2(
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
?
?
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873714

inputs
dense_28_1873703
dense_28_1873705
dense_29_1873708
dense_29_1873710
identity?? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCallinputsdense_28_1873703dense_28_1873705*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_18736202"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_1873708dense_29_1873710*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_18736662"
 dense_29/StatefulPartitionedCall?
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
E__inference_dense_29_layer_call_and_return_conditional_losses_1873985

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
?
?
/__inference_sequential_14_layer_call_fn_1873915

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
J__inference_sequential_14_layer_call_and_return_conditional_losses_18737412
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
/__inference_sequential_14_layer_call_fn_1873902

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
J__inference_sequential_14_layer_call_and_return_conditional_losses_18737142
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
/__inference_sequential_14_layer_call_fn_1873752
input_15
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2*
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
J__inference_sequential_14_layer_call_and_return_conditional_losses_18737412
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
input_15
?Z
?

#__inference__traced_restore_1874153
file_prefix$
 assignvariableop_dense_28_kernel$
 assignvariableop_1_dense_28_bias&
"assignvariableop_2_dense_29_kernel$
 assignvariableop_3_dense_29_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_1.
*assignvariableop_13_adam_dense_28_kernel_m,
(assignvariableop_14_adam_dense_28_bias_m.
*assignvariableop_15_adam_dense_29_kernel_m,
(assignvariableop_16_adam_dense_29_bias_m.
*assignvariableop_17_adam_dense_28_kernel_v,
(assignvariableop_18_adam_dense_28_bias_v.
*assignvariableop_19_adam_dense_29_kernel_v,
(assignvariableop_20_adam_dense_29_bias_v
identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_28_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_28_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_29_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_29_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_dense_28_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_dense_28_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_29_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_29_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_28_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_28_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_29_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_29_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21?
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873683
input_15
dense_28_1873631
dense_28_1873633
dense_29_1873677
dense_29_1873679
identity?? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCallinput_15dense_28_1873631dense_28_1873633*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_18736202"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_1873677dense_29_1873679*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_18736662"
 dense_29/StatefulPartitionedCall?
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_15
?W
?
"__inference__wrapped_model_1873585
input_15<
8sequential_14_dense_28_tensordot_readvariableop_resource:
6sequential_14_dense_28_biasadd_readvariableop_resource<
8sequential_14_dense_29_tensordot_readvariableop_resource:
6sequential_14_dense_29_biasadd_readvariableop_resource
identity??
/sequential_14/dense_28/Tensordot/ReadVariableOpReadVariableOp8sequential_14_dense_28_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_14/dense_28/Tensordot/ReadVariableOp?
%sequential_14/dense_28/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_14/dense_28/Tensordot/axes?
%sequential_14/dense_28/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_14/dense_28/Tensordot/free?
&sequential_14/dense_28/Tensordot/ShapeShapeinput_15*
T0*
_output_shapes
:2(
&sequential_14/dense_28/Tensordot/Shape?
.sequential_14/dense_28/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_14/dense_28/Tensordot/GatherV2/axis?
)sequential_14/dense_28/Tensordot/GatherV2GatherV2/sequential_14/dense_28/Tensordot/Shape:output:0.sequential_14/dense_28/Tensordot/free:output:07sequential_14/dense_28/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_14/dense_28/Tensordot/GatherV2?
0sequential_14/dense_28/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_14/dense_28/Tensordot/GatherV2_1/axis?
+sequential_14/dense_28/Tensordot/GatherV2_1GatherV2/sequential_14/dense_28/Tensordot/Shape:output:0.sequential_14/dense_28/Tensordot/axes:output:09sequential_14/dense_28/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_14/dense_28/Tensordot/GatherV2_1?
&sequential_14/dense_28/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_14/dense_28/Tensordot/Const?
%sequential_14/dense_28/Tensordot/ProdProd2sequential_14/dense_28/Tensordot/GatherV2:output:0/sequential_14/dense_28/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_14/dense_28/Tensordot/Prod?
(sequential_14/dense_28/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_14/dense_28/Tensordot/Const_1?
'sequential_14/dense_28/Tensordot/Prod_1Prod4sequential_14/dense_28/Tensordot/GatherV2_1:output:01sequential_14/dense_28/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_14/dense_28/Tensordot/Prod_1?
,sequential_14/dense_28/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_14/dense_28/Tensordot/concat/axis?
'sequential_14/dense_28/Tensordot/concatConcatV2.sequential_14/dense_28/Tensordot/free:output:0.sequential_14/dense_28/Tensordot/axes:output:05sequential_14/dense_28/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_14/dense_28/Tensordot/concat?
&sequential_14/dense_28/Tensordot/stackPack.sequential_14/dense_28/Tensordot/Prod:output:00sequential_14/dense_28/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_14/dense_28/Tensordot/stack?
*sequential_14/dense_28/Tensordot/transpose	Transposeinput_150sequential_14/dense_28/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22,
*sequential_14/dense_28/Tensordot/transpose?
(sequential_14/dense_28/Tensordot/ReshapeReshape.sequential_14/dense_28/Tensordot/transpose:y:0/sequential_14/dense_28/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_14/dense_28/Tensordot/Reshape?
'sequential_14/dense_28/Tensordot/MatMulMatMul1sequential_14/dense_28/Tensordot/Reshape:output:07sequential_14/dense_28/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_14/dense_28/Tensordot/MatMul?
(sequential_14/dense_28/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_14/dense_28/Tensordot/Const_2?
.sequential_14/dense_28/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_14/dense_28/Tensordot/concat_1/axis?
)sequential_14/dense_28/Tensordot/concat_1ConcatV22sequential_14/dense_28/Tensordot/GatherV2:output:01sequential_14/dense_28/Tensordot/Const_2:output:07sequential_14/dense_28/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_14/dense_28/Tensordot/concat_1?
 sequential_14/dense_28/TensordotReshape1sequential_14/dense_28/Tensordot/MatMul:product:02sequential_14/dense_28/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2"
 sequential_14/dense_28/Tensordot?
-sequential_14/dense_28/BiasAdd/ReadVariableOpReadVariableOp6sequential_14_dense_28_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_14/dense_28/BiasAdd/ReadVariableOp?
sequential_14/dense_28/BiasAddBiasAdd)sequential_14/dense_28/Tensordot:output:05sequential_14/dense_28/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2 
sequential_14/dense_28/BiasAdd?
sequential_14/dense_28/ReluRelu'sequential_14/dense_28/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
sequential_14/dense_28/Relu?
/sequential_14/dense_29/Tensordot/ReadVariableOpReadVariableOp8sequential_14_dense_29_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_14/dense_29/Tensordot/ReadVariableOp?
%sequential_14/dense_29/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_14/dense_29/Tensordot/axes?
%sequential_14/dense_29/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_14/dense_29/Tensordot/free?
&sequential_14/dense_29/Tensordot/ShapeShape)sequential_14/dense_28/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_14/dense_29/Tensordot/Shape?
.sequential_14/dense_29/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_14/dense_29/Tensordot/GatherV2/axis?
)sequential_14/dense_29/Tensordot/GatherV2GatherV2/sequential_14/dense_29/Tensordot/Shape:output:0.sequential_14/dense_29/Tensordot/free:output:07sequential_14/dense_29/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_14/dense_29/Tensordot/GatherV2?
0sequential_14/dense_29/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_14/dense_29/Tensordot/GatherV2_1/axis?
+sequential_14/dense_29/Tensordot/GatherV2_1GatherV2/sequential_14/dense_29/Tensordot/Shape:output:0.sequential_14/dense_29/Tensordot/axes:output:09sequential_14/dense_29/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_14/dense_29/Tensordot/GatherV2_1?
&sequential_14/dense_29/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_14/dense_29/Tensordot/Const?
%sequential_14/dense_29/Tensordot/ProdProd2sequential_14/dense_29/Tensordot/GatherV2:output:0/sequential_14/dense_29/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_14/dense_29/Tensordot/Prod?
(sequential_14/dense_29/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_14/dense_29/Tensordot/Const_1?
'sequential_14/dense_29/Tensordot/Prod_1Prod4sequential_14/dense_29/Tensordot/GatherV2_1:output:01sequential_14/dense_29/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_14/dense_29/Tensordot/Prod_1?
,sequential_14/dense_29/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_14/dense_29/Tensordot/concat/axis?
'sequential_14/dense_29/Tensordot/concatConcatV2.sequential_14/dense_29/Tensordot/free:output:0.sequential_14/dense_29/Tensordot/axes:output:05sequential_14/dense_29/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_14/dense_29/Tensordot/concat?
&sequential_14/dense_29/Tensordot/stackPack.sequential_14/dense_29/Tensordot/Prod:output:00sequential_14/dense_29/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_14/dense_29/Tensordot/stack?
*sequential_14/dense_29/Tensordot/transpose	Transpose)sequential_14/dense_28/Relu:activations:00sequential_14/dense_29/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2,
*sequential_14/dense_29/Tensordot/transpose?
(sequential_14/dense_29/Tensordot/ReshapeReshape.sequential_14/dense_29/Tensordot/transpose:y:0/sequential_14/dense_29/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_14/dense_29/Tensordot/Reshape?
'sequential_14/dense_29/Tensordot/MatMulMatMul1sequential_14/dense_29/Tensordot/Reshape:output:07sequential_14/dense_29/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_14/dense_29/Tensordot/MatMul?
(sequential_14/dense_29/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_14/dense_29/Tensordot/Const_2?
.sequential_14/dense_29/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_14/dense_29/Tensordot/concat_1/axis?
)sequential_14/dense_29/Tensordot/concat_1ConcatV22sequential_14/dense_29/Tensordot/GatherV2:output:01sequential_14/dense_29/Tensordot/Const_2:output:07sequential_14/dense_29/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_14/dense_29/Tensordot/concat_1?
 sequential_14/dense_29/TensordotReshape1sequential_14/dense_29/Tensordot/MatMul:product:02sequential_14/dense_29/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22"
 sequential_14/dense_29/Tensordot?
-sequential_14/dense_29/BiasAdd/ReadVariableOpReadVariableOp6sequential_14_dense_29_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_14/dense_29/BiasAdd/ReadVariableOp?
sequential_14/dense_29/BiasAddBiasAdd)sequential_14/dense_29/Tensordot:output:05sequential_14/dense_29/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22 
sequential_14/dense_29/BiasAdd
IdentityIdentity'sequential_14/dense_29/BiasAdd:output:0*
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
input_15
?

*__inference_dense_29_layer_call_fn_1873994

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
E__inference_dense_29_layer_call_and_return_conditional_losses_18736662
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
?
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873741

inputs
dense_28_1873730
dense_28_1873732
dense_29_1873735
dense_29_1873737
identity?? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCallinputsdense_28_1873730dense_28_1873732*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_18736202"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_1873735dense_29_1873737*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_18736662"
 dense_29/StatefulPartitionedCall?
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
/__inference_sequential_14_layer_call_fn_1873725
input_15
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_15unknown	unknown_0	unknown_1	unknown_2*
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
J__inference_sequential_14_layer_call_and_return_conditional_losses_18737142
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
input_15
?
?
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873697
input_15
dense_28_1873686
dense_28_1873688
dense_29_1873691
dense_29_1873693
identity?? dense_28/StatefulPartitionedCall? dense_29/StatefulPartitionedCall?
 dense_28/StatefulPartitionedCallStatefulPartitionedCallinput_15dense_28_1873686dense_28_1873688*
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
E__inference_dense_28_layer_call_and_return_conditional_losses_18736202"
 dense_28/StatefulPartitionedCall?
 dense_29/StatefulPartitionedCallStatefulPartitionedCall)dense_28/StatefulPartitionedCall:output:0dense_29_1873691dense_29_1873693*
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
E__inference_dense_29_layer_call_and_return_conditional_losses_18736662"
 dense_29/StatefulPartitionedCall?
IdentityIdentity)dense_29/StatefulPartitionedCall:output:0!^dense_28/StatefulPartitionedCall!^dense_29/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_28/StatefulPartitionedCall dense_28/StatefulPartitionedCall2D
 dense_29/StatefulPartitionedCall dense_29/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_15
?
?
E__inference_dense_28_layer_call_and_return_conditional_losses_1873946

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
input_155
serving_default_input_15:0?????????2@
dense_294
StatefulPartitionedCall:0?????????2tensorflow/serving/predict:?g
?
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
<_default_save_signature
=__call__
*>&call_and_return_all_conditional_losses"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_14", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_14", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_15"}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_14", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_15"}}, {"class_name": "Dense", "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

	kernel

bias
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_28", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_29", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 32]}}
?
iter

beta_1

beta_2
	decay
learning_rate	m4
m5m6m7	v8
v9v:v;"
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
non_trainable_variables
	variables

layers
regularization_losses
metrics
layer_metrics
layer_regularization_losses
=__call__
<_default_save_signature
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
,
Cserving_default"
signature_map
!: 2dense_28/kernel
: 2dense_28/bias
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
non_trainable_variables
	variables

 layers
regularization_losses
!metrics
"layer_metrics
#layer_regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
!: 2dense_29/kernel
:2dense_29/bias
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
$non_trainable_variables
	variables

%layers
regularization_losses
&metrics
'layer_metrics
(layer_regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
)0
*1"
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
?
	+total
	,count
-	variables
.	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	/total
	0count
1
_fn_kwargs
2	variables
3	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
+0
,1"
trackable_list_wrapper
-
-	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
-
2	variables"
_generic_user_object
&:$ 2Adam/dense_28/kernel/m
 : 2Adam/dense_28/bias/m
&:$ 2Adam/dense_29/kernel/m
 :2Adam/dense_29/bias/m
&:$ 2Adam/dense_28/kernel/v
 : 2Adam/dense_28/bias/v
&:$ 2Adam/dense_29/kernel/v
 :2Adam/dense_29/bias/v
?2?
"__inference__wrapped_model_1873585?
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
input_15?????????2
?2?
/__inference_sequential_14_layer_call_fn_1873902
/__inference_sequential_14_layer_call_fn_1873725
/__inference_sequential_14_layer_call_fn_1873915
/__inference_sequential_14_layer_call_fn_1873752?
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
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873697
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873832
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873683
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873889?
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
*__inference_dense_28_layer_call_fn_1873955?
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
E__inference_dense_28_layer_call_and_return_conditional_losses_1873946?
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
*__inference_dense_29_layer_call_fn_1873994?
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
E__inference_dense_29_layer_call_and_return_conditional_losses_1873985?
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
%__inference_signature_wrapper_1873775input_15?
"__inference__wrapped_model_1873585v	
5?2
+?(
&?#
input_15?????????2
? "7?4
2
dense_29&?#
dense_29?????????2?
E__inference_dense_28_layer_call_and_return_conditional_losses_1873946d	
3?0
)?&
$?!
inputs?????????2
? ")?&
?
0?????????2 
? ?
*__inference_dense_28_layer_call_fn_1873955W	
3?0
)?&
$?!
inputs?????????2
? "??????????2 ?
E__inference_dense_29_layer_call_and_return_conditional_losses_1873985d3?0
)?&
$?!
inputs?????????2 
? ")?&
?
0?????????2
? ?
*__inference_dense_29_layer_call_fn_1873994W3?0
)?&
$?!
inputs?????????2 
? "??????????2?
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873683p	
=?:
3?0
&?#
input_15?????????2
p

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873697p	
=?:
3?0
&?#
input_15?????????2
p 

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873832n	
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
J__inference_sequential_14_layer_call_and_return_conditional_losses_1873889n	
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
/__inference_sequential_14_layer_call_fn_1873725c	
=?:
3?0
&?#
input_15?????????2
p

 
? "??????????2?
/__inference_sequential_14_layer_call_fn_1873752c	
=?:
3?0
&?#
input_15?????????2
p 

 
? "??????????2?
/__inference_sequential_14_layer_call_fn_1873902a	
;?8
1?.
$?!
inputs?????????2
p

 
? "??????????2?
/__inference_sequential_14_layer_call_fn_1873915a	
;?8
1?.
$?!
inputs?????????2
p 

 
? "??????????2?
%__inference_signature_wrapper_1873775?	
A?>
? 
7?4
2
input_15&?#
input_15?????????2"7?4
2
dense_29&?#
dense_29?????????2