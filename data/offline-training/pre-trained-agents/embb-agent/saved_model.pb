ھ
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
�
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8�
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
�
5ActorDistributionNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense/kernel
�
IActorDistributionNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense/kernel*
_output_shapes

:	*
dtype0
�
3ActorDistributionNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53ActorDistributionNetwork/EncodingNetwork/dense/bias
�
GActorDistributionNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp3ActorDistributionNetwork/EncodingNetwork/dense/bias*
_output_shapes
:*
dtype0
�
7ActorDistributionNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*H
shared_name97ActorDistributionNetwork/EncodingNetwork/dense_1/kernel
�
KActorDistributionNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel*
_output_shapes

:*
dtype0
�
5ActorDistributionNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75ActorDistributionNetwork/EncodingNetwork/dense_1/bias
�
IActorDistributionNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp5ActorDistributionNetwork/EncodingNetwork/dense_1/bias*
_output_shapes
:*
dtype0
�
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*T
shared_nameECActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernel
�
WActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpReadVariableOpCActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernel*
_output_shapes
:	�*
dtype0
�
AActorDistributionNetwork/CategoricalProjectionNetwork/logits/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*R
shared_nameCAActorDistributionNetwork/CategoricalProjectionNetwork/logits/bias
�
UActorDistributionNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpReadVariableOpAActorDistributionNetwork/CategoricalProjectionNetwork/logits/bias*
_output_shapes	
:�*
dtype0
b
avg_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameavg_0
[
avg_0/Read/ReadVariableOpReadVariableOpavg_0*
_output_shapes
:	*
dtype0
f
count_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name	count_0
_
count_0/Read/ReadVariableOpReadVariableOpcount_0*
_output_shapes
:	*
dtype0
`
m2_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namem2_0
Y
m2_0/Read/ReadVariableOpReadVariableOpm2_0*
_output_shapes
:	*
dtype0
b
var_0VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namevar_0
[
var_0/Read/ReadVariableOpReadVariableOpvar_0*
_output_shapes
:	*
dtype0
�
+ValueNetwork/EncodingNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*<
shared_name-+ValueNetwork/EncodingNetwork/dense_2/kernel
�
?ValueNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOp+ValueNetwork/EncodingNetwork/dense_2/kernel*
_output_shapes

:	*
dtype0
�
)ValueNetwork/EncodingNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)ValueNetwork/EncodingNetwork/dense_2/bias
�
=ValueNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOp)ValueNetwork/EncodingNetwork/dense_2/bias*
_output_shapes
:*
dtype0
�
+ValueNetwork/EncodingNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+ValueNetwork/EncodingNetwork/dense_3/kernel
�
?ValueNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOp+ValueNetwork/EncodingNetwork/dense_3/kernel*
_output_shapes

:*
dtype0
�
)ValueNetwork/EncodingNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*:
shared_name+)ValueNetwork/EncodingNetwork/dense_3/bias
�
=ValueNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOp)ValueNetwork/EncodingNetwork/dense_3/bias*
_output_shapes
:*
dtype0
�
ValueNetwork/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameValueNetwork/dense_4/kernel
�
/ValueNetwork/dense_4/kernel/Read/ReadVariableOpReadVariableOpValueNetwork/dense_4/kernel*
_output_shapes

:*
dtype0
�
ValueNetwork/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameValueNetwork/dense_4/bias
�
-ValueNetwork/dense_4/bias/Read/ReadVariableOpReadVariableOpValueNetwork/dense_4/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�4
value�4B�4 B�4
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
v
0
1
2
	3

4
5
6
7
8
9
10
11
12
13
14
15

0
 
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE3ActorDistributionNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE5ActorDistributionNetwork/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUECActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAActorDistributionNetwork/CategoricalProjectionNetwork/logits/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEavg_0,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEcount_0,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEm2_0,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEvar_0,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE+ValueNetwork/EncodingNetwork/dense_2/kernel-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE)ValueNetwork/EncodingNetwork/dense_2/bias-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE+ValueNetwork/EncodingNetwork/dense_3/kernel-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE)ValueNetwork/EncodingNetwork/dense_3/bias-model_variables/13/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEValueNetwork/dense_4/kernel-model_variables/14/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEValueNetwork/dense_4/bias-model_variables/15/.ATTRIBUTES/VARIABLE_VALUE

ref
1
E
_actor_network
_observation_normalizer
_value_network
z
_encoder
_projection_networks
trainable_variables
regularization_losses
	variables
 	keras_api
@
!_flat_tensor_spec

"_count
#_avg
$_m2
%_var
|
&_encoder
'_postprocessing_layers
(trainable_variables
)regularization_losses
*	variables
+	keras_api
n
,_postprocessing_layers
-trainable_variables
.regularization_losses
/	variables
0	keras_api
i
1_projection_layer
2trainable_variables
3regularization_losses
4	variables
5	keras_api
*
0
1
2
	3

4
5
 
*
0
1
2
	3

4
5
�
6layer_regularization_losses
trainable_variables
regularization_losses

7layers
8layer_metrics
	variables
9non_trainable_variables
:metrics
 

0

0

0

0
n
;_postprocessing_layers
<trainable_variables
=regularization_losses
>	variables
?	keras_api
h

kernel
bias
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
*
0
1
2
3
4
5
 
*
0
1
2
3
4
5
�
Dlayer_regularization_losses
(trainable_variables
)regularization_losses

Elayers
Flayer_metrics
*	variables
Gnon_trainable_variables
Hmetrics

I0
J1
K2

0
1
2
	3
 

0
1
2
	3
�
Llayer_regularization_losses
-trainable_variables
.regularization_losses

Mlayers
Nlayer_metrics
/	variables
Onon_trainable_variables
Pmetrics
h


kernel
bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api


0
1
 


0
1
�
Ulayer_regularization_losses
2trainable_variables
3regularization_losses

Vlayers
Wlayer_metrics
4	variables
Xnon_trainable_variables
Ymetrics
 

0
1
 
 
 

Z0
[1
\2

0
1
2
3
 

0
1
2
3
�
]layer_regularization_losses
<trainable_variables
=regularization_losses

^layers
_layer_metrics
>	variables
`non_trainable_variables
ametrics

0
1
 

0
1
�
blayer_regularization_losses
@trainable_variables
Aregularization_losses

clayers
dlayer_metrics
B	variables
enon_trainable_variables
fmetrics
 

&0
'1
 
 
 
R
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
h

kernel
bias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
h

kernel
	bias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
 

I0
J1
K2
 
 
 


0
1
 


0
1
�
slayer_regularization_losses
Qtrainable_variables
Rregularization_losses

tlayers
ulayer_metrics
S	variables
vnon_trainable_variables
wmetrics
 

10
 
 
 
R
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
h

kernel
bias
|trainable_variables
}regularization_losses
~	variables
	keras_api
l

kernel
bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
 

Z0
[1
\2
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
�
 �layer_regularization_losses
gtrainable_variables
hregularization_losses
�layers
�layer_metrics
i	variables
�non_trainable_variables
�metrics

0
1
 

0
1
�
 �layer_regularization_losses
ktrainable_variables
lregularization_losses
�layers
�layer_metrics
m	variables
�non_trainable_variables
�metrics

0
	1
 

0
	1
�
 �layer_regularization_losses
otrainable_variables
pregularization_losses
�layers
�layer_metrics
q	variables
�non_trainable_variables
�metrics
 
 
 
 
 
 
 
 
�
 �layer_regularization_losses
xtrainable_variables
yregularization_losses
�layers
�layer_metrics
z	variables
�non_trainable_variables
�metrics

0
1
 

0
1
�
 �layer_regularization_losses
|trainable_variables
}regularization_losses
�layers
�layer_metrics
~	variables
�non_trainable_variables
�metrics

0
1
 

0
1
�
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�layers
�layer_metrics
�	variables
�non_trainable_variables
�metrics
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
l
action_0/discountPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
w
action_0/observationPlaceholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
j
action_0/rewardPlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
m
action_0/step_typePlaceholder*#
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_typevar_0avg_05ActorDistributionNetwork/EncodingNetwork/dense/kernel3ActorDistributionNetwork/EncodingNetwork/dense/bias7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel5ActorDistributionNetwork/EncodingNetwork/dense_1/biasCActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernelAActorDistributionNetwork/CategoricalProjectionNetwork/logits/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_1465628
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_1465640
�
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_1465662
�
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_signature_wrapper_1465655
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpKActorDistributionNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpIActorDistributionNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpWActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernel/Read/ReadVariableOpUActorDistributionNetwork/CategoricalProjectionNetwork/logits/bias/Read/ReadVariableOpavg_0/Read/ReadVariableOpcount_0/Read/ReadVariableOpm2_0/Read/ReadVariableOpvar_0/Read/ReadVariableOp?ValueNetwork/EncodingNetwork/dense_2/kernel/Read/ReadVariableOp=ValueNetwork/EncodingNetwork/dense_2/bias/Read/ReadVariableOp?ValueNetwork/EncodingNetwork/dense_3/kernel/Read/ReadVariableOp=ValueNetwork/EncodingNetwork/dense_3/bias/Read/ReadVariableOp/ValueNetwork/dense_4/kernel/Read/ReadVariableOp-ValueNetwork/dense_4/bias/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *)
f$R"
 __inference__traced_save_1465943
�
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable5ActorDistributionNetwork/EncodingNetwork/dense/kernel3ActorDistributionNetwork/EncodingNetwork/dense/bias7ActorDistributionNetwork/EncodingNetwork/dense_1/kernel5ActorDistributionNetwork/EncodingNetwork/dense_1/biasCActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernelAActorDistributionNetwork/CategoricalProjectionNetwork/logits/biasavg_0count_0m2_0var_0+ValueNetwork/EncodingNetwork/dense_2/kernel)ValueNetwork/EncodingNetwork/dense_2/bias+ValueNetwork/EncodingNetwork/dense_3/kernel)ValueNetwork/EncodingNetwork/dense_3/biasValueNetwork/dense_4/kernelValueNetwork/dense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *,
f'R%
#__inference__traced_restore_1466004ؼ
�0
�	
 __inference__traced_save_1465943
file_prefix'
#savev2_variable_read_readvariableop	T
Psavev2_actordistributionnetwork_encodingnetwork_dense_kernel_read_readvariableopR
Nsavev2_actordistributionnetwork_encodingnetwork_dense_bias_read_readvariableopV
Rsavev2_actordistributionnetwork_encodingnetwork_dense_1_kernel_read_readvariableopT
Psavev2_actordistributionnetwork_encodingnetwork_dense_1_bias_read_readvariableopb
^savev2_actordistributionnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableop`
\savev2_actordistributionnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop$
 savev2_avg_0_read_readvariableop&
"savev2_count_0_read_readvariableop#
savev2_m2_0_read_readvariableop$
 savev2_var_0_read_readvariableopJ
Fsavev2_valuenetwork_encodingnetwork_dense_2_kernel_read_readvariableopH
Dsavev2_valuenetwork_encodingnetwork_dense_2_bias_read_readvariableopJ
Fsavev2_valuenetwork_encodingnetwork_dense_3_kernel_read_readvariableopH
Dsavev2_valuenetwork_encodingnetwork_dense_3_bias_read_readvariableop:
6savev2_valuenetwork_dense_4_kernel_read_readvariableop8
4savev2_valuenetwork_dense_4_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_kernel_read_readvariableopNsavev2_actordistributionnetwork_encodingnetwork_dense_bias_read_readvariableopRsavev2_actordistributionnetwork_encodingnetwork_dense_1_kernel_read_readvariableopPsavev2_actordistributionnetwork_encodingnetwork_dense_1_bias_read_readvariableop^savev2_actordistributionnetwork_categoricalprojectionnetwork_logits_kernel_read_readvariableop\savev2_actordistributionnetwork_categoricalprojectionnetwork_logits_bias_read_readvariableop savev2_avg_0_read_readvariableop"savev2_count_0_read_readvariableopsavev2_m2_0_read_readvariableop savev2_var_0_read_readvariableopFsavev2_valuenetwork_encodingnetwork_dense_2_kernel_read_readvariableopDsavev2_valuenetwork_encodingnetwork_dense_2_bias_read_readvariableopFsavev2_valuenetwork_encodingnetwork_dense_3_kernel_read_readvariableopDsavev2_valuenetwork_encodingnetwork_dense_3_bias_read_readvariableop6savev2_valuenetwork_dense_4_kernel_read_readvariableop4savev2_valuenetwork_dense_4_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : :	::::	�:�:	:	:	:	:	:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::%!

_output_shapes
:	�:!

_output_shapes	
:�: 

_output_shapes
:	: 	

_output_shapes
:	: 


_output_shapes
:	: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
3

__inference_<lambda>_1465376*
_input_shapes 
׉
�
)__inference_polymorphic_action_fn_1465583
	time_step
time_step_1
time_step_2
time_step_3N
Jnormalize_observations_normalize_normalized_tensor_readvariableop_resourceP
Lnormalize_observations_normalize_normalized_tensor_readvariableop_1_resourceQ
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resourceR
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource_
[actordistributionnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource`
\actordistributionnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource
identity��SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp�RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp�EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�Anormalize_observations/normalize/normalized_tensor/ReadVariableOp�Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1�
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpReadVariableOpJnormalize_observations_normalize_normalized_tensor_readvariableop_resource*
_output_shapes
:	*
dtype02C
Anormalize_observations/normalize/normalized_tensor/ReadVariableOp�
8normalize_observations/normalize/normalized_tensor/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2:
8normalize_observations/normalize/normalized_tensor/add/y�
6normalize_observations/normalize/normalized_tensor/addAddV2Inormalize_observations/normalize/normalized_tensor/ReadVariableOp:value:0Anormalize_observations/normalize/normalized_tensor/add/y:output:0*
T0*
_output_shapes
:	28
6normalize_observations/normalize/normalized_tensor/add�
8normalize_observations/normalize/normalized_tensor/RsqrtRsqrt:normalize_observations/normalize/normalized_tensor/add:z:0*
T0*
_output_shapes
:	2:
8normalize_observations/normalize/normalized_tensor/Rsqrt�
6normalize_observations/normalize/normalized_tensor/mulMultime_step_3<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*'
_output_shapes
:���������	28
6normalize_observations/normalize/normalized_tensor/mul�
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1ReadVariableOpLnormalize_observations_normalize_normalized_tensor_readvariableop_1_resource*
_output_shapes
:	*
dtype02E
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1�
6normalize_observations/normalize/normalized_tensor/NegNegKnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	28
6normalize_observations/normalize/normalized_tensor/Neg�
8normalize_observations/normalize/normalized_tensor/mul_1Mul:normalize_observations/normalize/normalized_tensor/Neg:y:0<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*
_output_shapes
:	2:
8normalize_observations/normalize/normalized_tensor/mul_1�
8normalize_observations/normalize/normalized_tensor/add_1AddV2:normalize_observations/normalize/normalized_tensor/mul:z:0<normalize_observations/normalize/normalized_tensor/mul_1:z:0*
T0*'
_output_shapes
:���������	2:
8normalize_observations/normalize/normalized_tensor/add_1�
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@2F
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y�
Bnormalize_observations/normalize/clipped_normalized_tensor/MinimumMinimum<normalize_observations/normalize/normalized_tensor/add_1:z:0Mnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y:output:0*
T0*'
_output_shapes
:���������	2D
Bnormalize_observations/normalize/clipped_normalized_tensor/Minimum�
<normalize_observations/normalize/clipped_normalized_tensor/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��2>
<normalize_observations/normalize/clipped_normalized_tensor/y�
:normalize_observations/normalize/clipped_normalized_tensorMaximumFnormalize_observations/normalize/clipped_normalized_tensor/Minimum:z:0Enormalize_observations/normalize/clipped_normalized_tensor/y:output:0*
T0*'
_output_shapes
:���������	2<
:normalize_observations/normalize/clipped_normalized_tensor�
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����	   28
6ActorDistributionNetwork/EncodingNetwork/flatten/Const�
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshape>normalize_observations/normalize/clipped_normalized_tensor:z:0?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������	2:
8ActorDistributionNetwork/EncodingNetwork/flatten/Reshape�
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul�
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd�
3ActorDistributionNetwork/EncodingNetwork/dense/TanhTanh?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������25
3ActorDistributionNetwork/EncodingNetwork/dense/Tanh�
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMul7ActorDistributionNetwork/EncodingNetwork/dense/Tanh:y:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul�
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd�
5ActorDistributionNetwork/EncodingNetwork/dense_1/TanhTanhAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Tanh�
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp[actordistributionnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02T
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp�
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMulMatMul9ActorDistributionNetwork/EncodingNetwork/dense_1/Tanh:y:0ZActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2E
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul�
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp\actordistributionnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02U
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp�
DActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddMActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0[ActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
DActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd�
CActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����)  2E
CActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shape�
=ActorDistributionNetwork/CategoricalProjectionNetwork/ReshapeReshapeMActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0LActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:����������2?
=ActorDistributionNetwork/CategoricalProjectionNetwork/Reshape�
WActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2Y
WActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimension�
MActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMaxArgMaxFActorDistributionNetwork/CategoricalProjectionNetwork/Reshape:output:0`ActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2O
MActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax�
KActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/CastCastVActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2M
KActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeOActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToOActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapeu
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B :�2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value�
IdentityIdentityclip_by_value:z:0T^ActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpS^ActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpF^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpB^normalize_observations/normalize/normalized_tensor/ReadVariableOpD^normalize_observations/normalize/normalized_tensor/ReadVariableOp_1*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������	::::::::2�
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpSActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2�
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpRActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp2�
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2�
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2�
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2�
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2�
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpAnormalize_observations/normalize/normalized_tensor/ReadVariableOp2�
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1:N J
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:NJ
#
_output_shapes
:���������
#
_user_specified_name	time_step:RN
'
_output_shapes
:���������	
#
_user_specified_name	time_step
�
=
+__inference_function_with_signature_1465635

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *.
f)R'
%__inference_get_initial_state_14656342
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
͉
�
)__inference_polymorphic_action_fn_1465734
	step_type

reward
discount
observationN
Jnormalize_observations_normalize_normalized_tensor_readvariableop_resourceP
Lnormalize_observations_normalize_normalized_tensor_readvariableop_1_resourceQ
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resourceR
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource_
[actordistributionnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource`
\actordistributionnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource
identity��SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp�RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp�EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�Anormalize_observations/normalize/normalized_tensor/ReadVariableOp�Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1�
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpReadVariableOpJnormalize_observations_normalize_normalized_tensor_readvariableop_resource*
_output_shapes
:	*
dtype02C
Anormalize_observations/normalize/normalized_tensor/ReadVariableOp�
8normalize_observations/normalize/normalized_tensor/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2:
8normalize_observations/normalize/normalized_tensor/add/y�
6normalize_observations/normalize/normalized_tensor/addAddV2Inormalize_observations/normalize/normalized_tensor/ReadVariableOp:value:0Anormalize_observations/normalize/normalized_tensor/add/y:output:0*
T0*
_output_shapes
:	28
6normalize_observations/normalize/normalized_tensor/add�
8normalize_observations/normalize/normalized_tensor/RsqrtRsqrt:normalize_observations/normalize/normalized_tensor/add:z:0*
T0*
_output_shapes
:	2:
8normalize_observations/normalize/normalized_tensor/Rsqrt�
6normalize_observations/normalize/normalized_tensor/mulMulobservation<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*'
_output_shapes
:���������	28
6normalize_observations/normalize/normalized_tensor/mul�
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1ReadVariableOpLnormalize_observations_normalize_normalized_tensor_readvariableop_1_resource*
_output_shapes
:	*
dtype02E
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1�
6normalize_observations/normalize/normalized_tensor/NegNegKnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	28
6normalize_observations/normalize/normalized_tensor/Neg�
8normalize_observations/normalize/normalized_tensor/mul_1Mul:normalize_observations/normalize/normalized_tensor/Neg:y:0<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*
_output_shapes
:	2:
8normalize_observations/normalize/normalized_tensor/mul_1�
8normalize_observations/normalize/normalized_tensor/add_1AddV2:normalize_observations/normalize/normalized_tensor/mul:z:0<normalize_observations/normalize/normalized_tensor/mul_1:z:0*
T0*'
_output_shapes
:���������	2:
8normalize_observations/normalize/normalized_tensor/add_1�
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@2F
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y�
Bnormalize_observations/normalize/clipped_normalized_tensor/MinimumMinimum<normalize_observations/normalize/normalized_tensor/add_1:z:0Mnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y:output:0*
T0*'
_output_shapes
:���������	2D
Bnormalize_observations/normalize/clipped_normalized_tensor/Minimum�
<normalize_observations/normalize/clipped_normalized_tensor/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��2>
<normalize_observations/normalize/clipped_normalized_tensor/y�
:normalize_observations/normalize/clipped_normalized_tensorMaximumFnormalize_observations/normalize/clipped_normalized_tensor/Minimum:z:0Enormalize_observations/normalize/clipped_normalized_tensor/y:output:0*
T0*'
_output_shapes
:���������	2<
:normalize_observations/normalize/clipped_normalized_tensor�
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����	   28
6ActorDistributionNetwork/EncodingNetwork/flatten/Const�
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshape>normalize_observations/normalize/clipped_normalized_tensor:z:0?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������	2:
8ActorDistributionNetwork/EncodingNetwork/flatten/Reshape�
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul�
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd�
3ActorDistributionNetwork/EncodingNetwork/dense/TanhTanh?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������25
3ActorDistributionNetwork/EncodingNetwork/dense/Tanh�
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMul7ActorDistributionNetwork/EncodingNetwork/dense/Tanh:y:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul�
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd�
5ActorDistributionNetwork/EncodingNetwork/dense_1/TanhTanhAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Tanh�
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp[actordistributionnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02T
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp�
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMulMatMul9ActorDistributionNetwork/EncodingNetwork/dense_1/Tanh:y:0ZActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2E
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul�
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp\actordistributionnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02U
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp�
DActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddMActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0[ActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
DActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd�
CActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����)  2E
CActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shape�
=ActorDistributionNetwork/CategoricalProjectionNetwork/ReshapeReshapeMActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0LActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:����������2?
=ActorDistributionNetwork/CategoricalProjectionNetwork/Reshape�
WActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2Y
WActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimension�
MActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMaxArgMaxFActorDistributionNetwork/CategoricalProjectionNetwork/Reshape:output:0`ActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2O
MActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax�
KActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/CastCastVActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2M
KActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeOActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToOActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapeu
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B :�2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value�
IdentityIdentityclip_by_value:z:0T^ActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpS^ActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpF^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpB^normalize_observations/normalize/normalized_tensor/ReadVariableOpD^normalize_observations/normalize/normalized_tensor/ReadVariableOp_1*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������	::::::::2�
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpSActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2�
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpRActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp2�
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2�
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2�
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2�
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2�
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpAnormalize_observations/normalize/normalized_tensor/ReadVariableOp2�
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������	
%
_user_specified_nameobservation
�

�
%__inference_signature_wrapper_1465628
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_function_with_signature_14656022
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������	:���������:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������	
'
_user_specified_name0/observation:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:PL
#
_output_shapes
:���������
%
_user_specified_name0/step_type
��
�
)__inference_polymorphic_action_fn_1465806
time_step_step_type
time_step_reward
time_step_discount
time_step_observationN
Jnormalize_observations_normalize_normalized_tensor_readvariableop_resourceP
Lnormalize_observations_normalize_normalized_tensor_readvariableop_1_resourceQ
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resourceR
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource_
[actordistributionnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource`
\actordistributionnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource
identity��SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp�RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp�EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�Anormalize_observations/normalize/normalized_tensor/ReadVariableOp�Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1�
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpReadVariableOpJnormalize_observations_normalize_normalized_tensor_readvariableop_resource*
_output_shapes
:	*
dtype02C
Anormalize_observations/normalize/normalized_tensor/ReadVariableOp�
8normalize_observations/normalize/normalized_tensor/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2:
8normalize_observations/normalize/normalized_tensor/add/y�
6normalize_observations/normalize/normalized_tensor/addAddV2Inormalize_observations/normalize/normalized_tensor/ReadVariableOp:value:0Anormalize_observations/normalize/normalized_tensor/add/y:output:0*
T0*
_output_shapes
:	28
6normalize_observations/normalize/normalized_tensor/add�
8normalize_observations/normalize/normalized_tensor/RsqrtRsqrt:normalize_observations/normalize/normalized_tensor/add:z:0*
T0*
_output_shapes
:	2:
8normalize_observations/normalize/normalized_tensor/Rsqrt�
6normalize_observations/normalize/normalized_tensor/mulMultime_step_observation<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*'
_output_shapes
:���������	28
6normalize_observations/normalize/normalized_tensor/mul�
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1ReadVariableOpLnormalize_observations_normalize_normalized_tensor_readvariableop_1_resource*
_output_shapes
:	*
dtype02E
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1�
6normalize_observations/normalize/normalized_tensor/NegNegKnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	28
6normalize_observations/normalize/normalized_tensor/Neg�
8normalize_observations/normalize/normalized_tensor/mul_1Mul:normalize_observations/normalize/normalized_tensor/Neg:y:0<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*
_output_shapes
:	2:
8normalize_observations/normalize/normalized_tensor/mul_1�
8normalize_observations/normalize/normalized_tensor/add_1AddV2:normalize_observations/normalize/normalized_tensor/mul:z:0<normalize_observations/normalize/normalized_tensor/mul_1:z:0*
T0*'
_output_shapes
:���������	2:
8normalize_observations/normalize/normalized_tensor/add_1�
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@2F
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y�
Bnormalize_observations/normalize/clipped_normalized_tensor/MinimumMinimum<normalize_observations/normalize/normalized_tensor/add_1:z:0Mnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y:output:0*
T0*'
_output_shapes
:���������	2D
Bnormalize_observations/normalize/clipped_normalized_tensor/Minimum�
<normalize_observations/normalize/clipped_normalized_tensor/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��2>
<normalize_observations/normalize/clipped_normalized_tensor/y�
:normalize_observations/normalize/clipped_normalized_tensorMaximumFnormalize_observations/normalize/clipped_normalized_tensor/Minimum:z:0Enormalize_observations/normalize/clipped_normalized_tensor/y:output:0*
T0*'
_output_shapes
:���������	2<
:normalize_observations/normalize/clipped_normalized_tensor�
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����	   28
6ActorDistributionNetwork/EncodingNetwork/flatten/Const�
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshape>normalize_observations/normalize/clipped_normalized_tensor:z:0?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������	2:
8ActorDistributionNetwork/EncodingNetwork/flatten/Reshape�
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul�
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd�
3ActorDistributionNetwork/EncodingNetwork/dense/TanhTanh?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������25
3ActorDistributionNetwork/EncodingNetwork/dense/Tanh�
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMul7ActorDistributionNetwork/EncodingNetwork/dense/Tanh:y:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul�
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd�
5ActorDistributionNetwork/EncodingNetwork/dense_1/TanhTanhAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Tanh�
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp[actordistributionnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02T
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp�
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMulMatMul9ActorDistributionNetwork/EncodingNetwork/dense_1/Tanh:y:0ZActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2E
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul�
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp\actordistributionnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02U
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp�
DActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddMActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0[ActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
DActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd�
CActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����)  2E
CActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shape�
=ActorDistributionNetwork/CategoricalProjectionNetwork/ReshapeReshapeMActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0LActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:����������2?
=ActorDistributionNetwork/CategoricalProjectionNetwork/Reshape�
WActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2Y
WActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimension�
MActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMaxArgMaxFActorDistributionNetwork/CategoricalProjectionNetwork/Reshape:output:0`ActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2O
MActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax�
KActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/CastCastVActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2M
KActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtol�
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape�
Deterministic_1/sample/ShapeShapeOActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Cast:y:0*
T0*
_output_shapes
:2
Deterministic_1/sample/Shape�
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1�
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const�
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0�
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis�
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat�
"Deterministic_1/sample/BroadcastToBroadcastToOActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Cast:y:0&Deterministic_1/sample/concat:output:0*
T0*'
_output_shapes
:���������2$
"Deterministic_1/sample/BroadcastTo�
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_1�
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack�
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1�
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2�
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice�
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis�
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1�
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*#
_output_shapes
:���������2 
Deterministic_1/sample/Reshapeu
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value
B :�2
clip_by_value/Minimum/y�
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/y�
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:���������2
clip_by_value�
IdentityIdentityclip_by_value:z:0T^ActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpS^ActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpF^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpB^normalize_observations/normalize/normalized_tensor/ReadVariableOpD^normalize_observations/normalize/normalized_tensor/ReadVariableOp_1*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������	::::::::2�
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpSActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2�
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpRActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp2�
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2�
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2�
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2�
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2�
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpAnormalize_observations/normalize/normalized_tensor/ReadVariableOp2�
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1:X T
#
_output_shapes
:���������
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:���������
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:���������
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:���������	
/
_user_specified_nametime_step/observation
�
_
%__inference_signature_wrapper_1465655
unknown
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_function_with_signature_14656472
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
�
7
%__inference_get_initial_state_1465634

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
'
%__inference_signature_wrapper_1465662�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_function_with_signature_14656582
PartitionedCall*
_input_shapes 
�
7
%__inference_signature_wrapper_1465640

batch_size�
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *4
f/R-
+__inference_function_with_signature_14656352
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
]
__inference_<lambda>_1465373
readvariableop_resource
identity	��ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2 
ReadVariableOpReadVariableOp
�
7
%__inference_get_initial_state_1465864

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
�
-
+__inference_function_with_signature_1465658�
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *%
f R
__inference_<lambda>_14653762
PartitionedCall*
_input_shapes 
�o
�
/__inference_polymorphic_distribution_fn_1465861
	step_type

reward
discount
observationN
Jnormalize_observations_normalize_normalized_tensor_readvariableop_resourceP
Lnormalize_observations_normalize_normalized_tensor_readvariableop_1_resourceQ
Mactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resourceR
Nactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resourceS
Oactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resourceT
Pactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource_
[actordistributionnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource`
\actordistributionnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource
identity��SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp�RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp�EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�Anormalize_observations/normalize/normalized_tensor/ReadVariableOp�Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1�
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpReadVariableOpJnormalize_observations_normalize_normalized_tensor_readvariableop_resource*
_output_shapes
:	*
dtype02C
Anormalize_observations/normalize/normalized_tensor/ReadVariableOp�
8normalize_observations/normalize/normalized_tensor/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:2:
8normalize_observations/normalize/normalized_tensor/add/y�
6normalize_observations/normalize/normalized_tensor/addAddV2Inormalize_observations/normalize/normalized_tensor/ReadVariableOp:value:0Anormalize_observations/normalize/normalized_tensor/add/y:output:0*
T0*
_output_shapes
:	28
6normalize_observations/normalize/normalized_tensor/add�
8normalize_observations/normalize/normalized_tensor/RsqrtRsqrt:normalize_observations/normalize/normalized_tensor/add:z:0*
T0*
_output_shapes
:	2:
8normalize_observations/normalize/normalized_tensor/Rsqrt�
6normalize_observations/normalize/normalized_tensor/mulMulobservation<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*'
_output_shapes
:���������	28
6normalize_observations/normalize/normalized_tensor/mul�
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1ReadVariableOpLnormalize_observations_normalize_normalized_tensor_readvariableop_1_resource*
_output_shapes
:	*
dtype02E
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1�
6normalize_observations/normalize/normalized_tensor/NegNegKnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	28
6normalize_observations/normalize/normalized_tensor/Neg�
8normalize_observations/normalize/normalized_tensor/mul_1Mul:normalize_observations/normalize/normalized_tensor/Neg:y:0<normalize_observations/normalize/normalized_tensor/Rsqrt:y:0*
T0*
_output_shapes
:	2:
8normalize_observations/normalize/normalized_tensor/mul_1�
8normalize_observations/normalize/normalized_tensor/add_1AddV2:normalize_observations/normalize/normalized_tensor/mul:z:0<normalize_observations/normalize/normalized_tensor/mul_1:z:0*
T0*'
_output_shapes
:���������	2:
8normalize_observations/normalize/normalized_tensor/add_1�
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  �@2F
Dnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y�
Bnormalize_observations/normalize/clipped_normalized_tensor/MinimumMinimum<normalize_observations/normalize/normalized_tensor/add_1:z:0Mnormalize_observations/normalize/clipped_normalized_tensor/Minimum/y:output:0*
T0*'
_output_shapes
:���������	2D
Bnormalize_observations/normalize/clipped_normalized_tensor/Minimum�
<normalize_observations/normalize/clipped_normalized_tensor/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ��2>
<normalize_observations/normalize/clipped_normalized_tensor/y�
:normalize_observations/normalize/clipped_normalized_tensorMaximumFnormalize_observations/normalize/clipped_normalized_tensor/Minimum:z:0Enormalize_observations/normalize/clipped_normalized_tensor/y:output:0*
T0*'
_output_shapes
:���������	2<
:normalize_observations/normalize/clipped_normalized_tensor�
6ActorDistributionNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����	   28
6ActorDistributionNetwork/EncodingNetwork/flatten/Const�
8ActorDistributionNetwork/EncodingNetwork/flatten/ReshapeReshape>normalize_observations/normalize/clipped_normalized_tensor:z:0?ActorDistributionNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:���������	2:
8ActorDistributionNetwork/EncodingNetwork/flatten/Reshape�
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMactordistributionnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02F
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp�
5ActorDistributionNetwork/EncodingNetwork/dense/MatMulMatMulAActorDistributionNetwork/EncodingNetwork/flatten/Reshape:output:0LActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������27
5ActorDistributionNetwork/EncodingNetwork/dense/MatMul�
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNactordistributionnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp�
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAddBiasAdd?ActorDistributionNetwork/EncodingNetwork/dense/MatMul:product:0MActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������28
6ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd�
3ActorDistributionNetwork/EncodingNetwork/dense/TanhTanh?ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:���������25
3ActorDistributionNetwork/EncodingNetwork/dense/Tanh�
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpOactordistributionnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02H
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp�
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMulMatMul7ActorDistributionNetwork/EncodingNetwork/dense/Tanh:y:0NActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������29
7ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul�
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpPactordistributionnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02I
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp�
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAddBiasAddAActorDistributionNetwork/EncodingNetwork/dense_1/MatMul:product:0OActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2:
8ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd�
5ActorDistributionNetwork/EncodingNetwork/dense_1/TanhTanhAActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������27
5ActorDistributionNetwork/EncodingNetwork/dense_1/Tanh�
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpReadVariableOp[actordistributionnetwork_categoricalprojectionnetwork_logits_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype02T
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp�
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMulMatMul9ActorDistributionNetwork/EncodingNetwork/dense_1/Tanh:y:0ZActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2E
CActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul�
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpReadVariableOp\actordistributionnetwork_categoricalprojectionnetwork_logits_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02U
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp�
DActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAddBiasAddMActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul:product:0[ActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2F
DActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd�
CActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����)  2E
CActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shape�
=ActorDistributionNetwork/CategoricalProjectionNetwork/ReshapeReshapeMActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd:output:0LActorDistributionNetwork/CategoricalProjectionNetwork/Reshape/shape:output:0*
T0*(
_output_shapes
:����������2?
=ActorDistributionNetwork/CategoricalProjectionNetwork/Reshape�
WActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
���������2Y
WActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimension�
MActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMaxArgMaxFActorDistributionNetwork/CategoricalProjectionNetwork/Reshape:output:0`ActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:���������2O
MActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax�
KActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/CastCastVActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:���������2M
KActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Castj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_1/rtol�
IdentityIdentityOActorDistributionNetwork_CategoricalProjectionNetwork_Categorical/mode/Cast:y:0T^ActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpS^ActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpF^ActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^ActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpH^ActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpG^ActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpB^normalize_observations/normalize/normalized_tensor/ReadVariableOpD^normalize_observations/normalize/normalized_tensor/ReadVariableOp_1*
T0*#
_output_shapes
:���������2

Identityn
Deterministic_2/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic_2/rtol"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������	::::::::2�
SActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOpSActorDistributionNetwork/CategoricalProjectionNetwork/logits/BiasAdd/ReadVariableOp2�
RActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOpRActorDistributionNetwork/CategoricalProjectionNetwork/logits/MatMul/ReadVariableOp2�
EActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEActorDistributionNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2�
DActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDActorDistributionNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2�
GActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpGActorDistributionNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2�
FActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpFActorDistributionNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2�
Anormalize_observations/normalize/normalized_tensor/ReadVariableOpAnormalize_observations/normalize/normalized_tensor/ReadVariableOp2�
Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1Cnormalize_observations/normalize/normalized_tensor/ReadVariableOp_1:N J
#
_output_shapes
:���������
#
_user_specified_name	step_type:KG
#
_output_shapes
:���������
 
_user_specified_namereward:MI
#
_output_shapes
:���������
"
_user_specified_name
discount:TP
'
_output_shapes
:���������	
%
_user_specified_nameobservation
�M
�
#__inference__traced_restore_1466004
file_prefix
assignvariableop_variableL
Hassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_kernelJ
Fassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_biasN
Jassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_1_kernelL
Hassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_1_biasZ
Vassignvariableop_5_actordistributionnetwork_categoricalprojectionnetwork_logits_kernelX
Tassignvariableop_6_actordistributionnetwork_categoricalprojectionnetwork_logits_bias
assignvariableop_7_avg_0
assignvariableop_8_count_0
assignvariableop_9_m2_0
assignvariableop_10_var_0C
?assignvariableop_11_valuenetwork_encodingnetwork_dense_2_kernelA
=assignvariableop_12_valuenetwork_encodingnetwork_dense_2_biasC
?assignvariableop_13_valuenetwork_encodingnetwork_dense_3_kernelA
=assignvariableop_14_valuenetwork_encodingnetwork_dense_3_bias3
/assignvariableop_15_valuenetwork_dense_4_kernel1
-assignvariableop_16_valuenetwork_dense_4_bias
identity_18��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/13/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/14/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/15/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpHassignvariableop_1_actordistributionnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpFassignvariableop_2_actordistributionnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpJassignvariableop_3_actordistributionnetwork_encodingnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpHassignvariableop_4_actordistributionnetwork_encodingnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpVassignvariableop_5_actordistributionnetwork_categoricalprojectionnetwork_logits_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpTassignvariableop_6_actordistributionnetwork_categoricalprojectionnetwork_logits_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_avg_0Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_count_0Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_m2_0Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_var_0Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp?assignvariableop_11_valuenetwork_encodingnetwork_dense_2_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp=assignvariableop_12_valuenetwork_encodingnetwork_dense_2_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp?assignvariableop_13_valuenetwork_encodingnetwork_dense_3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp=assignvariableop_14_valuenetwork_encodingnetwork_dense_3_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_valuenetwork_dense_4_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp-assignvariableop_16_valuenetwork_dense_4_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_17�
Identity_18IdentityIdentity_17:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_18"#
identity_18Identity_18:output:0*Y
_input_shapesH
F: :::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
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
�
e
+__inference_function_with_signature_1465647
unknown
identity	��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8� *%
f R
__inference_<lambda>_14653732
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:22
StatefulPartitionedCallStatefulPartitionedCall
�

�
+__inference_function_with_signature_1465602
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:���������**
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8� *2
f-R+
)__inference_polymorphic_action_fn_14655832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:���������2

Identity"
identityIdentity:output:0*s
_input_shapesb
`:���������:���������:���������:���������	::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:���������
%
_user_specified_name0/step_type:MI
#
_output_shapes
:���������
"
_user_specified_name
0/reward:OK
#
_output_shapes
:���������
$
_user_specified_name
0/discount:VR
'
_output_shapes
:���������	
'
_user_specified_name0/observation"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
action�
4

0/discount&
action_0/discount:0���������
>
0/observation-
action_0/observation:0���������	
0
0/reward$
action_0/reward:0���������
6
0/step_type'
action_0/step_type:0���������6
action,
StatefulPartitionedCall:0���������tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:��
�

train_step
metadata
model_variables
_all_assets

signatures
�action
�distribution
�get_initial_state
�get_metadata
�get_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
�
0
1
2
	3

4
5
6
7
8
9
10
11
12
13
14
15"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
d
�action
�get_initial_state
�get_train_step
�get_metadata"
signature_map
G:E	25ActorDistributionNetwork/EncodingNetwork/dense/kernel
A:?23ActorDistributionNetwork/EncodingNetwork/dense/bias
I:G27ActorDistributionNetwork/EncodingNetwork/dense_1/kernel
C:A25ActorDistributionNetwork/EncodingNetwork/dense_1/bias
V:T	�2CActorDistributionNetwork/CategoricalProjectionNetwork/logits/kernel
P:N�2AActorDistributionNetwork/CategoricalProjectionNetwork/logits/bias
:	2avg_0
:	2count_0
:	2m2_0
:	2var_0
=:;	2+ValueNetwork/EncodingNetwork/dense_2/kernel
7:52)ValueNetwork/EncodingNetwork/dense_2/bias
=:;2+ValueNetwork/EncodingNetwork/dense_3/kernel
7:52)ValueNetwork/EncodingNetwork/dense_3/bias
-:+2ValueNetwork/dense_4/kernel
':%2ValueNetwork/dense_4/bias
1
ref
1"
trackable_tuple_wrapper
c
_actor_network
_observation_normalizer
_value_network"
_generic_user_object
�
_encoder
_projection_networks
trainable_variables
regularization_losses
	variables
 	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ActorDistributionNetwork", "name": "ActorDistributionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
^
!_flat_tensor_spec

"_count
#_avg
$_m2
%_var"
_generic_user_object
�
&_encoder
'_postprocessing_layers
(trainable_variables
)regularization_losses
*	variables
+	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ValueNetwork", "name": "ValueNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
,_postprocessing_layers
-trainable_variables
.regularization_losses
/	variables
0	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
1_projection_layer
2trainable_variables
3regularization_losses
4	variables
5	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "CategoricalProjectionNetwork", "name": "CategoricalProjectionNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
J
0
1
2
	3

4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
	3

4
5"
trackable_list_wrapper
�
6layer_regularization_losses
trainable_variables
regularization_losses

7layers
8layer_metrics
	variables
9non_trainable_variables
:metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
;_postprocessing_layers
<trainable_variables
=regularization_losses
>	variables
?	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�

kernel
bias
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
�
Dlayer_regularization_losses
(trainable_variables
)regularization_losses

Elayers
Flayer_metrics
*	variables
Gnon_trainable_variables
Hmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
5
I0
J1
K2"
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
	3"
trackable_list_wrapper
�
Llayer_regularization_losses
-trainable_variables
.regularization_losses

Mlayers
Nlayer_metrics
/	variables
Onon_trainable_variables
Pmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�


kernel
bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "logits", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "logits", "trainable": true, "dtype": "float32", "units": 297, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 0.1, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 30}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 30]}}
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
�
Ulayer_regularization_losses
2trainable_variables
3regularization_losses

Vlayers
Wlayer_metrics
4	variables
Xnon_trainable_variables
Ymetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
Z0
[1
\2"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
�
]layer_regularization_losses
<trainable_variables
=regularization_losses

^layers
_layer_metrics
>	variables
`non_trainable_variables
ametrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
blayer_regularization_losses
@trainable_variables
Aregularization_losses

clayers
dlayer_metrics
B	variables
enon_trainable_variables
fmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
gtrainable_variables
hregularization_losses
i	variables
j	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
ktrainable_variables
lregularization_losses
m	variables
n	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 5, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 9]}}
�

kernel
	bias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 5]}}
 "
trackable_list_wrapper
5
I0
J1
K2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
�
slayer_regularization_losses
Qtrainable_variables
Rregularization_losses

tlayers
ulayer_metrics
S	variables
vnon_trainable_variables
wmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
xtrainable_variables
yregularization_losses
z	variables
{	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

kernel
bias
|trainable_variables
}regularization_losses
~	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 5, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 9]}}
�

kernel
bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 30, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 5]}}
 "
trackable_list_wrapper
5
Z0
[1
\2"
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
trackable_list_wrapper
�
 �layer_regularization_losses
gtrainable_variables
hregularization_losses
�layers
�layer_metrics
i	variables
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 �layer_regularization_losses
ktrainable_variables
lregularization_losses
�layers
�layer_metrics
m	variables
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
�
 �layer_regularization_losses
otrainable_variables
pregularization_losses
�layers
�layer_metrics
q	variables
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
trackable_list_wrapper
�
 �layer_regularization_losses
xtrainable_variables
yregularization_losses
�layers
�layer_metrics
z	variables
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 �layer_regularization_losses
|trainable_variables
}regularization_losses
�layers
�layer_metrics
~	variables
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 �layer_regularization_losses
�trainable_variables
�regularization_losses
�layers
�layer_metrics
�	variables
�non_trainable_variables
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
�2�
)__inference_polymorphic_action_fn_1465734
)__inference_polymorphic_action_fn_1465806�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_polymorphic_distribution_fn_1465861�
���
FullArgSpec(
args �
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults�
� 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
%__inference_get_initial_state_1465864�
���
FullArgSpec!
args�
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_1465376"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_<lambda>_1465373"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1465628
0/discount0/observation0/reward0/step_type"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1465640
batch_size"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1465655"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_1465662"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpecU
argsM�J
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecU
argsM�J
jself
jobservations
j	step_type
jnetwork_state

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec?
args7�4
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec?
args7�4
jself
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpecL
argsD�A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults�

 
� 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ;
__inference_<lambda>_1465373�

� 
� "� 	4
__inference_<lambda>_1465376�

� 
� "� R
%__inference_get_initial_state_1465864)"�
�
�

batch_size 
� "� �
)__inference_polymorphic_action_fn_1465734�	
���
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������	
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
)__inference_polymorphic_action_fn_1465806�	
���
���
���
TimeStep6
	step_type)�&
time_step/step_type���������0
reward&�#
time_step/reward���������4
discount(�%
time_step/discount���������>
observation/�,
time_step/observation���������	
� 
� "R�O

PolicyStep&
action�
action���������
state� 
info� �
/__inference_polymorphic_distribution_fn_1465861�	
���
���
���
TimeStep,
	step_type�
	step_type���������&
reward�
reward���������*
discount�
discount���������4
observation%�"
observation���������	
� 
� "���

PolicyStep�
action�����Ã}�z
`
C�@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*�'
%
loc�
Identity���������
� _TFPTypeSpec
state� 
info� �
%__inference_signature_wrapper_1465628�	
���
� 
���
.

0/discount �

0/discount���������
8
0/observation'�$
0/observation���������	
*
0/reward�
0/reward���������
0
0/step_type!�
0/step_type���������"+�(
&
action�
action���������`
%__inference_signature_wrapper_146564070�-
� 
&�#
!

batch_size�

batch_size "� Y
%__inference_signature_wrapper_14656550�

� 
� "�

int64�
int64 	=
%__inference_signature_wrapper_1465662�

� 
� "� 