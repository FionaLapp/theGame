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
Adam/dense_32/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_32/kernel/m
?
*Adam/dense_32/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_32/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_32/bias/m
y
(Adam/dense_32/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_33/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_33/kernel/m
?
*Adam/dense_33/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/m*
_output_shapes

: *
dtype0
?
Adam/dense_33/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/m
y
(Adam/dense_33/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_32/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_32/kernel/v
?
*Adam/dense_32/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_32/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/dense_32/bias/v
y
(Adam/dense_32/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_32/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_33/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *'
shared_nameAdam/dense_33/kernel/v
?
*Adam/dense_33/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/kernel/v*
_output_shapes

: *
dtype0
?
Adam/dense_33/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_33/bias/v
y
(Adam/dense_33/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_33/bias/v*
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
non_trainable_variables
	variables

 layers
regularization_losses
!metrics
"layer_metrics
#layer_regularization_losses
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
VARIABLE_VALUEAdam/dense_32/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_32/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_32/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_33/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_33/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
%__inference_signature_wrapper_2103344
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_32/kernel/Read/ReadVariableOp!dense_32/bias/Read/ReadVariableOp#dense_33/kernel/Read/ReadVariableOp!dense_33/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_32/kernel/m/Read/ReadVariableOp(Adam/dense_32/bias/m/Read/ReadVariableOp*Adam/dense_33/kernel/m/Read/ReadVariableOp(Adam/dense_33/bias/m/Read/ReadVariableOp*Adam/dense_32/kernel/v/Read/ReadVariableOp(Adam/dense_32/bias/v/Read/ReadVariableOp*Adam/dense_33/kernel/v/Read/ReadVariableOp(Adam/dense_33/bias/v/Read/ReadVariableOpConst*"
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
 __inference__traced_save_2103649
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_32/kerneldense_32/biasdense_33/kerneldense_33/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_32/kernel/mAdam/dense_32/bias/mAdam/dense_33/kernel/mAdam/dense_33/bias/mAdam/dense_32/kernel/vAdam/dense_32/bias/vAdam/dense_33/kernel/vAdam/dense_33/bias/v*!
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
#__inference__traced_restore_2103722??
?
?
/__inference_sequential_16_layer_call_fn_2103471

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
J__inference_sequential_16_layer_call_and_return_conditional_losses_21032832
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103310

inputs
dense_32_2103299
dense_32_2103301
dense_33_2103304
dense_33_2103306
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_2103299dense_32_2103301*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_21031892"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_2103304dense_33_2103306*
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
E__inference_dense_33_layer_call_and_return_conditional_losses_21032352"
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
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103283

inputs
dense_32_2103272
dense_32_2103274
dense_33_2103277
dense_33_2103279
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinputsdense_32_2103272dense_32_2103274*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_21031892"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_2103277dense_33_2103279*
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
E__inference_dense_33_layer_call_and_return_conditional_losses_21032352"
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
/__inference_sequential_16_layer_call_fn_2103321
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_21033102
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
?

*__inference_dense_32_layer_call_fn_2103524

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
E__inference_dense_32_layer_call_and_return_conditional_losses_21031892
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
?Z
?

#__inference__traced_restore_2103722
file_prefix$
 assignvariableop_dense_32_kernel$
 assignvariableop_1_dense_32_bias&
"assignvariableop_2_dense_33_kernel$
 assignvariableop_3_dense_33_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_1.
*assignvariableop_13_adam_dense_32_kernel_m,
(assignvariableop_14_adam_dense_32_bias_m.
*assignvariableop_15_adam_dense_33_kernel_m,
(assignvariableop_16_adam_dense_33_bias_m.
*assignvariableop_17_adam_dense_32_kernel_v,
(assignvariableop_18_adam_dense_32_bias_v.
*assignvariableop_19_adam_dense_33_kernel_v,
(assignvariableop_20_adam_dense_33_bias_v
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
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_dense_32_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_dense_32_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_33_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_33_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_32_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_32_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_33_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_33_bias_vIdentity_20:output:0"/device:CPU:0*
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
?3
?
 __inference__traced_save_2103649
file_prefix.
*savev2_dense_32_kernel_read_readvariableop,
(savev2_dense_32_bias_read_readvariableop.
*savev2_dense_33_kernel_read_readvariableop,
(savev2_dense_33_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_32_kernel_m_read_readvariableop3
/savev2_adam_dense_32_bias_m_read_readvariableop5
1savev2_adam_dense_33_kernel_m_read_readvariableop3
/savev2_adam_dense_33_bias_m_read_readvariableop5
1savev2_adam_dense_32_kernel_v_read_readvariableop3
/savev2_adam_dense_32_bias_v_read_readvariableop5
1savev2_adam_dense_33_kernel_v_read_readvariableop3
/savev2_adam_dense_33_bias_v_read_readvariableop
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
value3B1 B+_temp_86e0bb6651a9485bb7c0f9e2d3da3147/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_32_kernel_read_readvariableop(savev2_dense_32_bias_read_readvariableop*savev2_dense_33_kernel_read_readvariableop(savev2_dense_33_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_32_kernel_m_read_readvariableop/savev2_adam_dense_32_bias_m_read_readvariableop1savev2_adam_dense_33_kernel_m_read_readvariableop/savev2_adam_dense_33_bias_m_read_readvariableop1savev2_adam_dense_32_kernel_v_read_readvariableop/savev2_adam_dense_32_bias_v_read_readvariableop1savev2_adam_dense_33_kernel_v_read_readvariableop/savev2_adam_dense_33_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?E
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103458

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
E__inference_dense_32_layer_call_and_return_conditional_losses_2103515

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
?
E__inference_dense_33_layer_call_and_return_conditional_losses_2103554

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
%__inference_signature_wrapper_2103344
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
"__inference__wrapped_model_21031542
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
?
?
/__inference_sequential_16_layer_call_fn_2103484

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
J__inference_sequential_16_layer_call_and_return_conditional_losses_21033102
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103252
input_17
dense_32_2103200
dense_32_2103202
dense_33_2103246
dense_33_2103248
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinput_17dense_32_2103200dense_32_2103202*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_21031892"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_2103246dense_33_2103248*
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
E__inference_dense_33_layer_call_and_return_conditional_losses_21032352"
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
*__inference_dense_33_layer_call_fn_2103563

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
E__inference_dense_33_layer_call_and_return_conditional_losses_21032352
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103266
input_17
dense_32_2103255
dense_32_2103257
dense_33_2103260
dense_33_2103262
identity?? dense_32/StatefulPartitionedCall? dense_33/StatefulPartitionedCall?
 dense_32/StatefulPartitionedCallStatefulPartitionedCallinput_17dense_32_2103255dense_32_2103257*
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
E__inference_dense_32_layer_call_and_return_conditional_losses_21031892"
 dense_32/StatefulPartitionedCall?
 dense_33/StatefulPartitionedCallStatefulPartitionedCall)dense_32/StatefulPartitionedCall:output:0dense_33_2103260dense_33_2103262*
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
E__inference_dense_33_layer_call_and_return_conditional_losses_21032352"
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
?W
?
"__inference__wrapped_model_2103154
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
?E
?
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103401

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
E__inference_dense_32_layer_call_and_return_conditional_losses_2103189

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
/__inference_sequential_16_layer_call_fn_2103294
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_21032832
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
?
E__inference_dense_33_layer_call_and_return_conditional_losses_2103235

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
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_16", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50, 6]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_17"}}, {"class_name": "Dense", "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mse", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
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
_tf_keras_layer?{"class_name": "Dense", "name": "dense_32", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_32", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 6]}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
A__call__
*B&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_33", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_33", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 32]}}
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
&:$ 2Adam/dense_32/kernel/m
 : 2Adam/dense_32/bias/m
&:$ 2Adam/dense_33/kernel/m
 :2Adam/dense_33/bias/m
&:$ 2Adam/dense_32/kernel/v
 : 2Adam/dense_32/bias/v
&:$ 2Adam/dense_33/kernel/v
 :2Adam/dense_33/bias/v
?2?
"__inference__wrapped_model_2103154?
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
/__inference_sequential_16_layer_call_fn_2103471
/__inference_sequential_16_layer_call_fn_2103294
/__inference_sequential_16_layer_call_fn_2103484
/__inference_sequential_16_layer_call_fn_2103321?
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103458
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103266
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103401
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103252?
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
*__inference_dense_32_layer_call_fn_2103524?
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
E__inference_dense_32_layer_call_and_return_conditional_losses_2103515?
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
*__inference_dense_33_layer_call_fn_2103563?
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
E__inference_dense_33_layer_call_and_return_conditional_losses_2103554?
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
%__inference_signature_wrapper_2103344input_17?
"__inference__wrapped_model_2103154v	
5?2
+?(
&?#
input_17?????????2
? "7?4
2
dense_33&?#
dense_33?????????2?
E__inference_dense_32_layer_call_and_return_conditional_losses_2103515d	
3?0
)?&
$?!
inputs?????????2
? ")?&
?
0?????????2 
? ?
*__inference_dense_32_layer_call_fn_2103524W	
3?0
)?&
$?!
inputs?????????2
? "??????????2 ?
E__inference_dense_33_layer_call_and_return_conditional_losses_2103554d3?0
)?&
$?!
inputs?????????2 
? ")?&
?
0?????????2
? ?
*__inference_dense_33_layer_call_fn_2103563W3?0
)?&
$?!
inputs?????????2 
? "??????????2?
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103252p	
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103266p	
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103401n	
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
J__inference_sequential_16_layer_call_and_return_conditional_losses_2103458n	
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
/__inference_sequential_16_layer_call_fn_2103294c	
=?:
3?0
&?#
input_17?????????2
p

 
? "??????????2?
/__inference_sequential_16_layer_call_fn_2103321c	
=?:
3?0
&?#
input_17?????????2
p 

 
? "??????????2?
/__inference_sequential_16_layer_call_fn_2103471a	
;?8
1?.
$?!
inputs?????????2
p

 
? "??????????2?
/__inference_sequential_16_layer_call_fn_2103484a	
;?8
1?.
$?!
inputs?????????2
p 

 
? "??????????2?
%__inference_signature_wrapper_2103344?	
A?>
? 
7?4
2
input_17&?#
input_17?????????2"7?4
2
dense_33&?#
dense_33?????????2