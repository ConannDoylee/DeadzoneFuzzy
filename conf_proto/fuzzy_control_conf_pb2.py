# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fuzzy_control_conf.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='fuzzy_control_conf.proto',
  package='control.fuzzy',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x18\x66uzzy_control_conf.proto\x12\rcontrol.fuzzy\"*\n\x0eSimulationConf\x12\t\n\x01T\x18\x01 \x01(\x01\x12\r\n\x05\x63ycle\x18\x02 \x01(\x05\"?\n\x12MemberFunctionConf\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\r\n\x05\x61rray\x18\x03 \x03(\x01\"U\n\x0e\x41ntecedentConf\x12.\n\x03mfs\x18\x01 \x03(\x0b\x32!.control.fuzzy.MemberFunctionConf\x12\x13\n\x0brange_array\x18\x02 \x03(\x01\"8\n\x08RuleConf\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x10\n\x08mf_names\x18\x03 \x03(\t\"\x94\x01\n\x0e\x46uzzyBuildConf\x12,\n\x05\x61nt_1\x18\x01 \x01(\x0b\x32\x1d.control.fuzzy.AntecedentConf\x12,\n\x05\x61nt_2\x18\x02 \x01(\x0b\x32\x1d.control.fuzzy.AntecedentConf\x12&\n\x05rules\x18\x03 \x03(\x0b\x32\x17.control.fuzzy.RuleConf\"\xb5\x01\n\x10\x43ompensationConf\x12\x37\n\x10\x66uzzy_build_conf\x18\x01 \x01(\x0b\x32\x1d.control.fuzzy.FuzzyBuildConf\x12\x0b\n\x03phi\x18\x02 \x01(\x01\x12\x1c\n\x14init_adaptive_values\x18\x03 \x03(\x01\x12\x1f\n\x17init_compensation_value\x18\x04 \x01(\x01\x12\x1c\n\x14use_compensation_out\x18\x05 \x01(\x08\"\\\n\x0e\x43ontrollerConf\x12\t\n\x01\x62\x18\x01 \x01(\x01\x12\t\n\x01m\x18\x02 \x01(\x01\x12\x0f\n\x07lambbda\x18\x03 \x01(\x01\x12\x0c\n\x04kesi\x18\x04 \x01(\x01\x12\t\n\x01n\x18\x05 \x01(\x01\x12\n\n\x02mu\x18\x06 \x01(\x01\";\n\x0c\x44\x65\x61\x64zoneConf\x12\t\n\x01m\x18\x01 \x01(\x01\x12\x0f\n\x07\x64\x65lta_l\x18\x02 \x01(\x01\x12\x0f\n\x07\x64\x65lta_r\x18\x03 \x01(\x01\"\x8b\x01\n\tModelConf\x12\t\n\x01N\x18\x01 \x01(\x01\x12\t\n\x01\x62\x18\x02 \x01(\x01\x12\x32\n\rdeadzone_conf\x18\x03 \x01(\x0b\x32\x1b.control.fuzzy.DeadzoneConf\x12\x12\n\ninit_value\x18\x04 \x03(\x01\x12\n\n\x02mu\x18\x05 \x01(\x01\x12\x14\n\x0cuse_deadzone\x18\x06 \x01(\x08'
)




_SIMULATIONCONF = _descriptor.Descriptor(
  name='SimulationConf',
  full_name='control.fuzzy.SimulationConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='T', full_name='control.fuzzy.SimulationConf.T', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cycle', full_name='control.fuzzy.SimulationConf.cycle', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=43,
  serialized_end=85,
)


_MEMBERFUNCTIONCONF = _descriptor.Descriptor(
  name='MemberFunctionConf',
  full_name='control.fuzzy.MemberFunctionConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='control.fuzzy.MemberFunctionConf.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='control.fuzzy.MemberFunctionConf.type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='array', full_name='control.fuzzy.MemberFunctionConf.array', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=87,
  serialized_end=150,
)


_ANTECEDENTCONF = _descriptor.Descriptor(
  name='AntecedentConf',
  full_name='control.fuzzy.AntecedentConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='mfs', full_name='control.fuzzy.AntecedentConf.mfs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='range_array', full_name='control.fuzzy.AntecedentConf.range_array', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=152,
  serialized_end=237,
)


_RULECONF = _descriptor.Descriptor(
  name='RuleConf',
  full_name='control.fuzzy.RuleConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='control.fuzzy.RuleConf.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='type', full_name='control.fuzzy.RuleConf.type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mf_names', full_name='control.fuzzy.RuleConf.mf_names', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=239,
  serialized_end=295,
)


_FUZZYBUILDCONF = _descriptor.Descriptor(
  name='FuzzyBuildConf',
  full_name='control.fuzzy.FuzzyBuildConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ant_1', full_name='control.fuzzy.FuzzyBuildConf.ant_1', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ant_2', full_name='control.fuzzy.FuzzyBuildConf.ant_2', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rules', full_name='control.fuzzy.FuzzyBuildConf.rules', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=298,
  serialized_end=446,
)


_COMPENSATIONCONF = _descriptor.Descriptor(
  name='CompensationConf',
  full_name='control.fuzzy.CompensationConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='fuzzy_build_conf', full_name='control.fuzzy.CompensationConf.fuzzy_build_conf', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='phi', full_name='control.fuzzy.CompensationConf.phi', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='init_adaptive_values', full_name='control.fuzzy.CompensationConf.init_adaptive_values', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='init_compensation_value', full_name='control.fuzzy.CompensationConf.init_compensation_value', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_compensation_out', full_name='control.fuzzy.CompensationConf.use_compensation_out', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=449,
  serialized_end=630,
)


_CONTROLLERCONF = _descriptor.Descriptor(
  name='ControllerConf',
  full_name='control.fuzzy.ControllerConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='b', full_name='control.fuzzy.ControllerConf.b', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='m', full_name='control.fuzzy.ControllerConf.m', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='lambbda', full_name='control.fuzzy.ControllerConf.lambbda', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='kesi', full_name='control.fuzzy.ControllerConf.kesi', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='n', full_name='control.fuzzy.ControllerConf.n', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mu', full_name='control.fuzzy.ControllerConf.mu', index=5,
      number=6, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=632,
  serialized_end=724,
)


_DEADZONECONF = _descriptor.Descriptor(
  name='DeadzoneConf',
  full_name='control.fuzzy.DeadzoneConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='m', full_name='control.fuzzy.DeadzoneConf.m', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='delta_l', full_name='control.fuzzy.DeadzoneConf.delta_l', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='delta_r', full_name='control.fuzzy.DeadzoneConf.delta_r', index=2,
      number=3, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=726,
  serialized_end=785,
)


_MODELCONF = _descriptor.Descriptor(
  name='ModelConf',
  full_name='control.fuzzy.ModelConf',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='N', full_name='control.fuzzy.ModelConf.N', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='b', full_name='control.fuzzy.ModelConf.b', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='deadzone_conf', full_name='control.fuzzy.ModelConf.deadzone_conf', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='init_value', full_name='control.fuzzy.ModelConf.init_value', index=3,
      number=4, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mu', full_name='control.fuzzy.ModelConf.mu', index=4,
      number=5, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_deadzone', full_name='control.fuzzy.ModelConf.use_deadzone', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=788,
  serialized_end=927,
)

_ANTECEDENTCONF.fields_by_name['mfs'].message_type = _MEMBERFUNCTIONCONF
_FUZZYBUILDCONF.fields_by_name['ant_1'].message_type = _ANTECEDENTCONF
_FUZZYBUILDCONF.fields_by_name['ant_2'].message_type = _ANTECEDENTCONF
_FUZZYBUILDCONF.fields_by_name['rules'].message_type = _RULECONF
_COMPENSATIONCONF.fields_by_name['fuzzy_build_conf'].message_type = _FUZZYBUILDCONF
_MODELCONF.fields_by_name['deadzone_conf'].message_type = _DEADZONECONF
DESCRIPTOR.message_types_by_name['SimulationConf'] = _SIMULATIONCONF
DESCRIPTOR.message_types_by_name['MemberFunctionConf'] = _MEMBERFUNCTIONCONF
DESCRIPTOR.message_types_by_name['AntecedentConf'] = _ANTECEDENTCONF
DESCRIPTOR.message_types_by_name['RuleConf'] = _RULECONF
DESCRIPTOR.message_types_by_name['FuzzyBuildConf'] = _FUZZYBUILDCONF
DESCRIPTOR.message_types_by_name['CompensationConf'] = _COMPENSATIONCONF
DESCRIPTOR.message_types_by_name['ControllerConf'] = _CONTROLLERCONF
DESCRIPTOR.message_types_by_name['DeadzoneConf'] = _DEADZONECONF
DESCRIPTOR.message_types_by_name['ModelConf'] = _MODELCONF
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

SimulationConf = _reflection.GeneratedProtocolMessageType('SimulationConf', (_message.Message,), {
  'DESCRIPTOR' : _SIMULATIONCONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.SimulationConf)
  })
_sym_db.RegisterMessage(SimulationConf)

MemberFunctionConf = _reflection.GeneratedProtocolMessageType('MemberFunctionConf', (_message.Message,), {
  'DESCRIPTOR' : _MEMBERFUNCTIONCONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.MemberFunctionConf)
  })
_sym_db.RegisterMessage(MemberFunctionConf)

AntecedentConf = _reflection.GeneratedProtocolMessageType('AntecedentConf', (_message.Message,), {
  'DESCRIPTOR' : _ANTECEDENTCONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.AntecedentConf)
  })
_sym_db.RegisterMessage(AntecedentConf)

RuleConf = _reflection.GeneratedProtocolMessageType('RuleConf', (_message.Message,), {
  'DESCRIPTOR' : _RULECONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.RuleConf)
  })
_sym_db.RegisterMessage(RuleConf)

FuzzyBuildConf = _reflection.GeneratedProtocolMessageType('FuzzyBuildConf', (_message.Message,), {
  'DESCRIPTOR' : _FUZZYBUILDCONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.FuzzyBuildConf)
  })
_sym_db.RegisterMessage(FuzzyBuildConf)

CompensationConf = _reflection.GeneratedProtocolMessageType('CompensationConf', (_message.Message,), {
  'DESCRIPTOR' : _COMPENSATIONCONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.CompensationConf)
  })
_sym_db.RegisterMessage(CompensationConf)

ControllerConf = _reflection.GeneratedProtocolMessageType('ControllerConf', (_message.Message,), {
  'DESCRIPTOR' : _CONTROLLERCONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.ControllerConf)
  })
_sym_db.RegisterMessage(ControllerConf)

DeadzoneConf = _reflection.GeneratedProtocolMessageType('DeadzoneConf', (_message.Message,), {
  'DESCRIPTOR' : _DEADZONECONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.DeadzoneConf)
  })
_sym_db.RegisterMessage(DeadzoneConf)

ModelConf = _reflection.GeneratedProtocolMessageType('ModelConf', (_message.Message,), {
  'DESCRIPTOR' : _MODELCONF,
  '__module__' : 'fuzzy_control_conf_pb2'
  # @@protoc_insertion_point(class_scope:control.fuzzy.ModelConf)
  })
_sym_db.RegisterMessage(ModelConf)


# @@protoc_insertion_point(module_scope)
