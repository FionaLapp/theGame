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
dense_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_36/kernel
s
#dense_36/kernel/Read/ReadVariableOpReadVariableOpdense_36/kernel*
_output_shapes

: *
dtype0
r
dense_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_36/bias
k
!dense_36/bias/Read/ReadVariableOpReadVariableOpdense_36/bias*
_output_shapes
: *
dtype0
z
dense_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: * 
shared_namedense_37/kernel
s
#dense_37/kernel/Read/ReadVariableOpReadVariableOpdense_37/kernel*
_output_shapes

: *
dtype0
r
dense_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_37/bias
k
!dense_37/bias/Read/ReadVariableOpReadVariableOpdense_37/bias*
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
Adam/dense_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_36/kernel/m
?
*Adam/dense_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_36/bias/m
y
(Adam/dense_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_37/kernel/m
?
*Adam/dense_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_37/bias/m
y
(Adam/dense_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_36/kernel/v
?
*Adam/dense_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_36/bias/v
y
(Adam/dense_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_36/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_37/kernel/v
?
*Adam/dense_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_37/bias/v
y
(Adam/dense_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_37/bias/v*
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
VARIABLE_VALUEdense_36/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_36/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_37/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_37/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_36/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_36/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_36/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_36/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_37/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_37/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_19Placeholder*+
_output_shapes
:?????????2*
dtype0* 
shape:?????????2
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_19dense_36/kerneldense_36/biasdense_37/kerneldense_37/bias*
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
%__inference_signature_wrapper_3117443
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_36/kernel/Read/ReadVariableOp!dense_36/bias/Read/ReadVariableOp#dense_37/kernel/Read/ReadVariableOp!dense_37/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_36/kernel/m/Read/ReadVariableOp(Adam/dense_36/bias/m/Read/ReadVariableOp*Adam/dense_37/kernel/m/Read/ReadVariableOp(Adam/dense_37/bias/m/Read/ReadVariableOp*Adam/dense_36/kernel/v/Read/ReadVariableOp(Adam/dense_36/bias/v/Read/ReadVariableOp*Adam/dense_37/kernel/v/Read/ReadVariableOp(Adam/dense_37/bias/v/Read/ReadVariableOpConst*"
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
 __inference__traced_save_3117748
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_36/kerneldense_36/biasdense_37/kerneldense_37/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_36/kernel/mAdam/dense_36/bias/mAdam/dense_37/kernel/mAdam/dense_37/bias/mAdam/dense_36/kernel/vAdam/dense_36/bias/vAdam/dense_37/kernel/vAdam/dense_37/bias/v*!
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
#__inference__traced_restore_3117821??
?
?
E__inference_dense_36_layer_call_and_return_conditional_losses_3117614

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
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117351
input_19
dense_36_3117299
dense_36_3117301
dense_37_3117345
dense_37_3117347
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_36_3117299dense_36_3117301*
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
E__inference_dense_36_layer_call_and_return_conditional_losses_31172882"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_3117345dense_37_3117347*
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
E__inference_dense_37_layer_call_and_return_conditional_losses_31173342"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_19
?
?
/__inference_sequential_18_layer_call_fn_3117570

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
J__inference_sequential_18_layer_call_and_return_conditional_losses_31173822
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
?Z
?

#__inference__traced_restore_3117821
file_prefix$
 assignvariableop_dense_36_kernel$
 assignvariableop_1_dense_36_bias&
"assignvariableop_2_dense_37_kernel$
 assignvariableop_3_dense_37_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_1.
*assignvariableop_13_adam_dense_36_kernel_m,
(assignvariableop_14_adam_dense_36_bias_m.
*assignvariableop_15_adam_dense_37_kernel_m,
(assignvariableop_16_adam_dense_37_bias_m.
*assignvariableop_17_adam_dense_36_kernel_v,
(assignvariableop_18_adam_dense_36_bias_v.
*assignvariableop_19_adam_dense_37_kernel_v,
(assignvariableop_20_adam_dense_37_bias_v
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
AssignVariableOpAssignVariableOp assignvariableop_dense_36_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_36_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_37_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_37_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_dense_36_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_dense_36_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_37_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_37_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_36_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_36_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_37_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_37_bias_vIdentity_20:output:0"/device:CPU:0*
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117365
input_19
dense_36_3117354
dense_36_3117356
dense_37_3117359
dense_37_3117361
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinput_19dense_36_3117354dense_36_3117356*
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
E__inference_dense_36_layer_call_and_return_conditional_losses_31172882"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_3117359dense_37_3117361*
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
E__inference_dense_37_layer_call_and_return_conditional_losses_31173342"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????2
"
_user_specified_name
input_19
?
?
/__inference_sequential_18_layer_call_fn_3117393
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2*
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_31173822
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
input_19
?
?
E__inference_dense_37_layer_call_and_return_conditional_losses_3117653

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
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117409

inputs
dense_36_3117398
dense_36_3117400
dense_37_3117403
dense_37_3117405
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_3117398dense_36_3117400*
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
E__inference_dense_36_layer_call_and_return_conditional_losses_31172882"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_3117403dense_37_3117405*
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
E__inference_dense_37_layer_call_and_return_conditional_losses_31173342"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?

*__inference_dense_37_layer_call_fn_3117662

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
E__inference_dense_37_layer_call_and_return_conditional_losses_31173342
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
E__inference_dense_36_layer_call_and_return_conditional_losses_3117288

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
?3
?
 __inference__traced_save_3117748
file_prefix.
*savev2_dense_36_kernel_read_readvariableop,
(savev2_dense_36_bias_read_readvariableop.
*savev2_dense_37_kernel_read_readvariableop,
(savev2_dense_37_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_36_kernel_m_read_readvariableop3
/savev2_adam_dense_36_bias_m_read_readvariableop5
1savev2_adam_dense_37_kernel_m_read_readvariableop3
/savev2_adam_dense_37_bias_m_read_readvariableop5
1savev2_adam_dense_36_kernel_v_read_readvariableop3
/savev2_adam_dense_36_bias_v_read_readvariableop5
1savev2_adam_dense_37_kernel_v_read_readvariableop3
/savev2_adam_dense_37_bias_v_read_readvariableop
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
value3B1 B+_temp_032297690fb84543863b7dbd9f7165e4/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_36_kernel_read_readvariableop(savev2_dense_36_bias_read_readvariableop*savev2_dense_37_kernel_read_readvariableop(savev2_dense_37_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_36_kernel_m_read_readvariableop/savev2_adam_dense_36_bias_m_read_readvariableop1savev2_adam_dense_37_kernel_m_read_readvariableop/savev2_adam_dense_37_bias_m_read_readvariableop1savev2_adam_dense_36_kernel_v_read_readvariableop/savev2_adam_dense_36_bias_v_read_readvariableop1savev2_adam_dense_37_kernel_v_read_readvariableop/savev2_adam_dense_37_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?

*__inference_dense_36_layer_call_fn_3117623

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
E__inference_dense_36_layer_call_and_return_conditional_losses_31172882
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117500

inputs.
*dense_36_tensordot_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource.
*dense_37_tensordot_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??
!dense_36/Tensordot/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_36/Tensordot/ReadVariableOp|
dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_36/Tensordot/axes?
dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_36/Tensordot/freej
dense_36/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_36/Tensordot/Shape?
 dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_36/Tensordot/GatherV2/axis?
dense_36/Tensordot/GatherV2GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/free:output:0)dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_36/Tensordot/GatherV2?
"dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_36/Tensordot/GatherV2_1/axis?
dense_36/Tensordot/GatherV2_1GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/axes:output:0+dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_36/Tensordot/GatherV2_1~
dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const?
dense_36/Tensordot/ProdProd$dense_36/Tensordot/GatherV2:output:0!dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_36/Tensordot/Prod?
dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const_1?
dense_36/Tensordot/Prod_1Prod&dense_36/Tensordot/GatherV2_1:output:0#dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/Tensordot/Prod_1?
dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_36/Tensordot/concat/axis?
dense_36/Tensordot/concatConcatV2 dense_36/Tensordot/free:output:0 dense_36/Tensordot/axes:output:0'dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/concat?
dense_36/Tensordot/stackPack dense_36/Tensordot/Prod:output:0"dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/stack?
dense_36/Tensordot/transpose	Transposeinputs"dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_36/Tensordot/transpose?
dense_36/Tensordot/ReshapeReshape dense_36/Tensordot/transpose:y:0!dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_36/Tensordot/Reshape?
dense_36/Tensordot/MatMulMatMul#dense_36/Tensordot/Reshape:output:0)dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_36/Tensordot/MatMul?
dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const_2?
 dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_36/Tensordot/concat_1/axis?
dense_36/Tensordot/concat_1ConcatV2$dense_36/Tensordot/GatherV2:output:0#dense_36/Tensordot/Const_2:output:0)dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/concat_1?
dense_36/TensordotReshape#dense_36/Tensordot/MatMul:product:0$dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_36/Tensordot?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/Tensordot:output:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_36/BiasAddw
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_36/Relu?
!dense_37/Tensordot/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_37/Tensordot/ReadVariableOp|
dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_37/Tensordot/axes?
dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_37/Tensordot/free
dense_37/Tensordot/ShapeShapedense_36/Relu:activations:0*
T0*
_output_shapes
:2
dense_37/Tensordot/Shape?
 dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/GatherV2/axis?
dense_37/Tensordot/GatherV2GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/free:output:0)dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2?
"dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_37/Tensordot/GatherV2_1/axis?
dense_37/Tensordot/GatherV2_1GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/axes:output:0+dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2_1~
dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const?
dense_37/Tensordot/ProdProd$dense_37/Tensordot/GatherV2:output:0!dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod?
dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const_1?
dense_37/Tensordot/Prod_1Prod&dense_37/Tensordot/GatherV2_1:output:0#dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod_1?
dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_37/Tensordot/concat/axis?
dense_37/Tensordot/concatConcatV2 dense_37/Tensordot/free:output:0 dense_37/Tensordot/axes:output:0'dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat?
dense_37/Tensordot/stackPack dense_37/Tensordot/Prod:output:0"dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/stack?
dense_37/Tensordot/transpose	Transposedense_36/Relu:activations:0"dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_37/Tensordot/transpose?
dense_37/Tensordot/ReshapeReshape dense_37/Tensordot/transpose:y:0!dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_37/Tensordot/Reshape?
dense_37/Tensordot/MatMulMatMul#dense_37/Tensordot/Reshape:output:0)dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/Tensordot/MatMul?
dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_37/Tensordot/Const_2?
 dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/concat_1/axis?
dense_37/Tensordot/concat_1ConcatV2$dense_37/Tensordot/GatherV2:output:0#dense_37/Tensordot/Const_2:output:0)dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat_1?
dense_37/TensordotReshape#dense_37/Tensordot/MatMul:product:0$dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_37/Tensordot?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/Tensordot:output:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_37/BiasAddq
IdentityIdentitydense_37/BiasAdd:output:0*
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
%__inference_signature_wrapper_3117443
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2*
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
"__inference__wrapped_model_31172532
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
input_19
?W
?
"__inference__wrapped_model_3117253
input_19<
8sequential_18_dense_36_tensordot_readvariableop_resource:
6sequential_18_dense_36_biasadd_readvariableop_resource<
8sequential_18_dense_37_tensordot_readvariableop_resource:
6sequential_18_dense_37_biasadd_readvariableop_resource
identity??
/sequential_18/dense_36/Tensordot/ReadVariableOpReadVariableOp8sequential_18_dense_36_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_18/dense_36/Tensordot/ReadVariableOp?
%sequential_18/dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_18/dense_36/Tensordot/axes?
%sequential_18/dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_18/dense_36/Tensordot/free?
&sequential_18/dense_36/Tensordot/ShapeShapeinput_19*
T0*
_output_shapes
:2(
&sequential_18/dense_36/Tensordot/Shape?
.sequential_18/dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_36/Tensordot/GatherV2/axis?
)sequential_18/dense_36/Tensordot/GatherV2GatherV2/sequential_18/dense_36/Tensordot/Shape:output:0.sequential_18/dense_36/Tensordot/free:output:07sequential_18/dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_18/dense_36/Tensordot/GatherV2?
0sequential_18/dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_18/dense_36/Tensordot/GatherV2_1/axis?
+sequential_18/dense_36/Tensordot/GatherV2_1GatherV2/sequential_18/dense_36/Tensordot/Shape:output:0.sequential_18/dense_36/Tensordot/axes:output:09sequential_18/dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_18/dense_36/Tensordot/GatherV2_1?
&sequential_18/dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_18/dense_36/Tensordot/Const?
%sequential_18/dense_36/Tensordot/ProdProd2sequential_18/dense_36/Tensordot/GatherV2:output:0/sequential_18/dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_18/dense_36/Tensordot/Prod?
(sequential_18/dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_18/dense_36/Tensordot/Const_1?
'sequential_18/dense_36/Tensordot/Prod_1Prod4sequential_18/dense_36/Tensordot/GatherV2_1:output:01sequential_18/dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_18/dense_36/Tensordot/Prod_1?
,sequential_18/dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_18/dense_36/Tensordot/concat/axis?
'sequential_18/dense_36/Tensordot/concatConcatV2.sequential_18/dense_36/Tensordot/free:output:0.sequential_18/dense_36/Tensordot/axes:output:05sequential_18/dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_18/dense_36/Tensordot/concat?
&sequential_18/dense_36/Tensordot/stackPack.sequential_18/dense_36/Tensordot/Prod:output:00sequential_18/dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_18/dense_36/Tensordot/stack?
*sequential_18/dense_36/Tensordot/transpose	Transposeinput_190sequential_18/dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22,
*sequential_18/dense_36/Tensordot/transpose?
(sequential_18/dense_36/Tensordot/ReshapeReshape.sequential_18/dense_36/Tensordot/transpose:y:0/sequential_18/dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_18/dense_36/Tensordot/Reshape?
'sequential_18/dense_36/Tensordot/MatMulMatMul1sequential_18/dense_36/Tensordot/Reshape:output:07sequential_18/dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2)
'sequential_18/dense_36/Tensordot/MatMul?
(sequential_18/dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_18/dense_36/Tensordot/Const_2?
.sequential_18/dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_36/Tensordot/concat_1/axis?
)sequential_18/dense_36/Tensordot/concat_1ConcatV22sequential_18/dense_36/Tensordot/GatherV2:output:01sequential_18/dense_36/Tensordot/Const_2:output:07sequential_18/dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_18/dense_36/Tensordot/concat_1?
 sequential_18/dense_36/TensordotReshape1sequential_18/dense_36/Tensordot/MatMul:product:02sequential_18/dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2"
 sequential_18/dense_36/Tensordot?
-sequential_18/dense_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_18/dense_36/BiasAdd/ReadVariableOp?
sequential_18/dense_36/BiasAddBiasAdd)sequential_18/dense_36/Tensordot:output:05sequential_18/dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2 
sequential_18/dense_36/BiasAdd?
sequential_18/dense_36/ReluRelu'sequential_18/dense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
sequential_18/dense_36/Relu?
/sequential_18/dense_37/Tensordot/ReadVariableOpReadVariableOp8sequential_18_dense_37_tensordot_readvariableop_resource*
_output_shapes

: *
dtype021
/sequential_18/dense_37/Tensordot/ReadVariableOp?
%sequential_18/dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_18/dense_37/Tensordot/axes?
%sequential_18/dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_18/dense_37/Tensordot/free?
&sequential_18/dense_37/Tensordot/ShapeShape)sequential_18/dense_36/Relu:activations:0*
T0*
_output_shapes
:2(
&sequential_18/dense_37/Tensordot/Shape?
.sequential_18/dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_37/Tensordot/GatherV2/axis?
)sequential_18/dense_37/Tensordot/GatherV2GatherV2/sequential_18/dense_37/Tensordot/Shape:output:0.sequential_18/dense_37/Tensordot/free:output:07sequential_18/dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_18/dense_37/Tensordot/GatherV2?
0sequential_18/dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_18/dense_37/Tensordot/GatherV2_1/axis?
+sequential_18/dense_37/Tensordot/GatherV2_1GatherV2/sequential_18/dense_37/Tensordot/Shape:output:0.sequential_18/dense_37/Tensordot/axes:output:09sequential_18/dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_18/dense_37/Tensordot/GatherV2_1?
&sequential_18/dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_18/dense_37/Tensordot/Const?
%sequential_18/dense_37/Tensordot/ProdProd2sequential_18/dense_37/Tensordot/GatherV2:output:0/sequential_18/dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_18/dense_37/Tensordot/Prod?
(sequential_18/dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_18/dense_37/Tensordot/Const_1?
'sequential_18/dense_37/Tensordot/Prod_1Prod4sequential_18/dense_37/Tensordot/GatherV2_1:output:01sequential_18/dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_18/dense_37/Tensordot/Prod_1?
,sequential_18/dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_18/dense_37/Tensordot/concat/axis?
'sequential_18/dense_37/Tensordot/concatConcatV2.sequential_18/dense_37/Tensordot/free:output:0.sequential_18/dense_37/Tensordot/axes:output:05sequential_18/dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_18/dense_37/Tensordot/concat?
&sequential_18/dense_37/Tensordot/stackPack.sequential_18/dense_37/Tensordot/Prod:output:00sequential_18/dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_18/dense_37/Tensordot/stack?
*sequential_18/dense_37/Tensordot/transpose	Transpose)sequential_18/dense_36/Relu:activations:00sequential_18/dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2,
*sequential_18/dense_37/Tensordot/transpose?
(sequential_18/dense_37/Tensordot/ReshapeReshape.sequential_18/dense_37/Tensordot/transpose:y:0/sequential_18/dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2*
(sequential_18/dense_37/Tensordot/Reshape?
'sequential_18/dense_37/Tensordot/MatMulMatMul1sequential_18/dense_37/Tensordot/Reshape:output:07sequential_18/dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'sequential_18/dense_37/Tensordot/MatMul?
(sequential_18/dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_18/dense_37/Tensordot/Const_2?
.sequential_18/dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_18/dense_37/Tensordot/concat_1/axis?
)sequential_18/dense_37/Tensordot/concat_1ConcatV22sequential_18/dense_37/Tensordot/GatherV2:output:01sequential_18/dense_37/Tensordot/Const_2:output:07sequential_18/dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_18/dense_37/Tensordot/concat_1?
 sequential_18/dense_37/TensordotReshape1sequential_18/dense_37/Tensordot/MatMul:product:02sequential_18/dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22"
 sequential_18/dense_37/Tensordot?
-sequential_18/dense_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_18_dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_18/dense_37/BiasAdd/ReadVariableOp?
sequential_18/dense_37/BiasAddBiasAdd)sequential_18/dense_37/Tensordot:output:05sequential_18/dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22 
sequential_18/dense_37/BiasAdd
IdentityIdentity'sequential_18/dense_37/BiasAdd:output:0*
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
input_19
?E
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117557

inputs.
*dense_36_tensordot_readvariableop_resource,
(dense_36_biasadd_readvariableop_resource.
*dense_37_tensordot_readvariableop_resource,
(dense_37_biasadd_readvariableop_resource
identity??
!dense_36/Tensordot/ReadVariableOpReadVariableOp*dense_36_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_36/Tensordot/ReadVariableOp|
dense_36/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_36/Tensordot/axes?
dense_36/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_36/Tensordot/freej
dense_36/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
dense_36/Tensordot/Shape?
 dense_36/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_36/Tensordot/GatherV2/axis?
dense_36/Tensordot/GatherV2GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/free:output:0)dense_36/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_36/Tensordot/GatherV2?
"dense_36/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_36/Tensordot/GatherV2_1/axis?
dense_36/Tensordot/GatherV2_1GatherV2!dense_36/Tensordot/Shape:output:0 dense_36/Tensordot/axes:output:0+dense_36/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_36/Tensordot/GatherV2_1~
dense_36/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const?
dense_36/Tensordot/ProdProd$dense_36/Tensordot/GatherV2:output:0!dense_36/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_36/Tensordot/Prod?
dense_36/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const_1?
dense_36/Tensordot/Prod_1Prod&dense_36/Tensordot/GatherV2_1:output:0#dense_36/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_36/Tensordot/Prod_1?
dense_36/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_36/Tensordot/concat/axis?
dense_36/Tensordot/concatConcatV2 dense_36/Tensordot/free:output:0 dense_36/Tensordot/axes:output:0'dense_36/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/concat?
dense_36/Tensordot/stackPack dense_36/Tensordot/Prod:output:0"dense_36/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/stack?
dense_36/Tensordot/transpose	Transposeinputs"dense_36/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????22
dense_36/Tensordot/transpose?
dense_36/Tensordot/ReshapeReshape dense_36/Tensordot/transpose:y:0!dense_36/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_36/Tensordot/Reshape?
dense_36/Tensordot/MatMulMatMul#dense_36/Tensordot/Reshape:output:0)dense_36/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense_36/Tensordot/MatMul?
dense_36/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_36/Tensordot/Const_2?
 dense_36/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_36/Tensordot/concat_1/axis?
dense_36/Tensordot/concat_1ConcatV2$dense_36/Tensordot/GatherV2:output:0#dense_36/Tensordot/Const_2:output:0)dense_36/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_36/Tensordot/concat_1?
dense_36/TensordotReshape#dense_36/Tensordot/MatMul:product:0$dense_36/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2 2
dense_36/Tensordot?
dense_36/BiasAdd/ReadVariableOpReadVariableOp(dense_36_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
dense_36/BiasAdd/ReadVariableOp?
dense_36/BiasAddBiasAdddense_36/Tensordot:output:0'dense_36/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2 2
dense_36/BiasAddw
dense_36/ReluReludense_36/BiasAdd:output:0*
T0*+
_output_shapes
:?????????2 2
dense_36/Relu?
!dense_37/Tensordot/ReadVariableOpReadVariableOp*dense_37_tensordot_readvariableop_resource*
_output_shapes

: *
dtype02#
!dense_37/Tensordot/ReadVariableOp|
dense_37/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_37/Tensordot/axes?
dense_37/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_37/Tensordot/free
dense_37/Tensordot/ShapeShapedense_36/Relu:activations:0*
T0*
_output_shapes
:2
dense_37/Tensordot/Shape?
 dense_37/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/GatherV2/axis?
dense_37/Tensordot/GatherV2GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/free:output:0)dense_37/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2?
"dense_37/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_37/Tensordot/GatherV2_1/axis?
dense_37/Tensordot/GatherV2_1GatherV2!dense_37/Tensordot/Shape:output:0 dense_37/Tensordot/axes:output:0+dense_37/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_37/Tensordot/GatherV2_1~
dense_37/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const?
dense_37/Tensordot/ProdProd$dense_37/Tensordot/GatherV2:output:0!dense_37/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod?
dense_37/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_37/Tensordot/Const_1?
dense_37/Tensordot/Prod_1Prod&dense_37/Tensordot/GatherV2_1:output:0#dense_37/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_37/Tensordot/Prod_1?
dense_37/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_37/Tensordot/concat/axis?
dense_37/Tensordot/concatConcatV2 dense_37/Tensordot/free:output:0 dense_37/Tensordot/axes:output:0'dense_37/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat?
dense_37/Tensordot/stackPack dense_37/Tensordot/Prod:output:0"dense_37/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/stack?
dense_37/Tensordot/transpose	Transposedense_36/Relu:activations:0"dense_37/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2 2
dense_37/Tensordot/transpose?
dense_37/Tensordot/ReshapeReshape dense_37/Tensordot/transpose:y:0!dense_37/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_37/Tensordot/Reshape?
dense_37/Tensordot/MatMulMatMul#dense_37/Tensordot/Reshape:output:0)dense_37/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_37/Tensordot/MatMul?
dense_37/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_37/Tensordot/Const_2?
 dense_37/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_37/Tensordot/concat_1/axis?
dense_37/Tensordot/concat_1ConcatV2$dense_37/Tensordot/GatherV2:output:0#dense_37/Tensordot/Const_2:output:0)dense_37/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_37/Tensordot/concat_1?
dense_37/TensordotReshape#dense_37/Tensordot/MatMul:product:0$dense_37/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????22
dense_37/Tensordot?
dense_37/BiasAdd/ReadVariableOpReadVariableOp(dense_37_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_37/BiasAdd/ReadVariableOp?
dense_37/BiasAddBiasAdddense_37/Tensordot:output:0'dense_37/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????22
dense_37/BiasAddq
IdentityIdentitydense_37/BiasAdd:output:0*
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
/__inference_sequential_18_layer_call_fn_3117420
input_19
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_19unknown	unknown_0	unknown_1	unknown_2*
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_31174092
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
input_19
?
?
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117382

inputs
dense_36_3117371
dense_36_3117373
dense_37_3117376
dense_37_3117378
identity?? dense_36/StatefulPartitionedCall? dense_37/StatefulPartitionedCall?
 dense_36/StatefulPartitionedCallStatefulPartitionedCallinputsdense_36_3117371dense_36_3117373*
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
E__inference_dense_36_layer_call_and_return_conditional_losses_31172882"
 dense_36/StatefulPartitionedCall?
 dense_37/StatefulPartitionedCallStatefulPartitionedCall)dense_36/StatefulPartitionedCall:output:0dense_37_3117376dense_37_3117378*
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
E__inference_dense_37_layer_call_and_return_conditional_losses_31173342"
 dense_37/StatefulPartitionedCall?
IdentityIdentity)dense_37/StatefulPartitionedCall:output:0!^dense_36/StatefulPartitionedCall!^dense_37/StatefulPartitionedCall*
T0*+
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*:
_input_shapes)
':?????????2::::2D
 dense_36/StatefulPartitionedCall dense_36/StatefulPartitionedCall2D
 dense_37/StatefulPartitionedCall dense_37/StatefulPartitionedCall:S O
+
_output_shapes
:?????????2
 
_user_specified_nameinputs
?
?
E__inference_dense_37_layer_call_and_return_conditional_losses_3117334

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
/__inference_sequential_18_layer_call_fn_3117583

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
J__inference_sequential_18_layer_call_and_return_conditional_losses_31174092
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
input_195
serving_default_input_19:0?????????2@
dense_374
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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_18", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_19"}}, {"class_name": "Dense", "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_36", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_37", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_37", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 32]}}
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
!: 2dense_36/kernel
: 2dense_36/bias
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
!: 2dense_37/kernel
:2dense_37/bias
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
&:$ 2Adam/dense_36/kernel/m
 : 2Adam/dense_36/bias/m
&:$ 2Adam/dense_37/kernel/m
 :2Adam/dense_37/bias/m
&:$ 2Adam/dense_36/kernel/v
 : 2Adam/dense_36/bias/v
&:$ 2Adam/dense_37/kernel/v
 :2Adam/dense_37/bias/v
?2?
"__inference__wrapped_model_3117253?
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
input_19?????????2
?2?
/__inference_sequential_18_layer_call_fn_3117583
/__inference_sequential_18_layer_call_fn_3117420
/__inference_sequential_18_layer_call_fn_3117393
/__inference_sequential_18_layer_call_fn_3117570?
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117557
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117500
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117365
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117351?
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
*__inference_dense_36_layer_call_fn_3117623?
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
E__inference_dense_36_layer_call_and_return_conditional_losses_3117614?
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
*__inference_dense_37_layer_call_fn_3117662?
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
E__inference_dense_37_layer_call_and_return_conditional_losses_3117653?
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
%__inference_signature_wrapper_3117443input_19?
"__inference__wrapped_model_3117253v	
5?2
+?(
&?#
input_19?????????2
? "7?4
2
dense_37&?#
dense_37?????????2?
E__inference_dense_36_layer_call_and_return_conditional_losses_3117614d	
3?0
)?&
$?!
inputs?????????2
? ")?&
?
0?????????2 
? ?
*__inference_dense_36_layer_call_fn_3117623W	
3?0
)?&
$?!
inputs?????????2
? "??????????2 ?
E__inference_dense_37_layer_call_and_return_conditional_losses_3117653d3?0
)?&
$?!
inputs?????????2 
? ")?&
?
0?????????2
? ?
*__inference_dense_37_layer_call_fn_3117662W3?0
)?&
$?!
inputs?????????2 
? "??????????2?
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117351p	
=?:
3?0
&?#
input_19?????????2
p

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117365p	
=?:
3?0
&?#
input_19?????????2
p 

 
? ")?&
?
0?????????2
? ?
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117500n	
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
J__inference_sequential_18_layer_call_and_return_conditional_losses_3117557n	
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
/__inference_sequential_18_layer_call_fn_3117393c	
=?:
3?0
&?#
input_19?????????2
p

 
? "??????????2?
/__inference_sequential_18_layer_call_fn_3117420c	
=?:
3?0
&?#
input_19?????????2
p 

 
? "??????????2?
/__inference_sequential_18_layer_call_fn_3117570a	
;?8
1?.
$?!
inputs?????????2
p

 
? "??????????2?
/__inference_sequential_18_layer_call_fn_3117583a	
;?8
1?.
$?!
inputs?????????2
p 

 
? "??????????2?
%__inference_signature_wrapper_3117443?	
A?>
? 
7?4
2
input_19&?#
input_19?????????2"7?4
2
dense_37&?#
dense_37?????????2