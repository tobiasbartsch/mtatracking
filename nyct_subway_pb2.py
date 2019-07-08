# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nyct-subway.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import mtatracking.gtfs_realtime_pb2 as gtfs__realtime__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='nyct-subway.proto',
  package='',
  syntax='proto2',
  serialized_options=_b('\n\033com.google.transit.realtime'),
  serialized_pb=_b('\n\x11nyct-subway.proto\x1a\x13gtfs-realtime.proto\"b\n\x15TripReplacementPeriod\x12\x10\n\x08route_id\x18\x01 \x01(\t\x12\x37\n\x12replacement_period\x18\x02 \x01(\x0b\x32\x1b.transit_realtime.TimeRange\"f\n\x0eNyctFeedHeader\x12\x1b\n\x13nyct_subway_version\x18\x01 \x02(\t\x12\x37\n\x17trip_replacement_period\x18\x02 \x03(\x0b\x32\x16.TripReplacementPeriod\"\xa4\x01\n\x12NyctTripDescriptor\x12\x10\n\x08train_id\x18\x01 \x01(\t\x12\x13\n\x0bis_assigned\x18\x02 \x01(\x08\x12\x30\n\tdirection\x18\x03 \x01(\x0e\x32\x1d.NyctTripDescriptor.Direction\"5\n\tDirection\x12\t\n\x05NORTH\x10\x01\x12\x08\n\x04\x45\x41ST\x10\x02\x12\t\n\x05SOUTH\x10\x03\x12\x08\n\x04WEST\x10\x04\"C\n\x12NyctStopTimeUpdate\x12\x17\n\x0fscheduled_track\x18\x01 \x01(\t\x12\x14\n\x0c\x61\x63tual_track\x18\x02 \x01(\t:H\n\x10nyct_feed_header\x12\x1c.transit_realtime.FeedHeader\x18\xe9\x07 \x01(\x0b\x32\x0f.NyctFeedHeader:T\n\x14nyct_trip_descriptor\x12 .transit_realtime.TripDescriptor\x18\xe9\x07 \x01(\x0b\x32\x13.NyctTripDescriptor:`\n\x15nyct_stop_time_update\x12+.transit_realtime.TripUpdate.StopTimeUpdate\x18\xe9\x07 \x01(\x0b\x32\x13.NyctStopTimeUpdateB\x1d\n\x1b\x63om.google.transit.realtime')
  ,
  dependencies=[gtfs__realtime__pb2.DESCRIPTOR,])


NYCT_FEED_HEADER_FIELD_NUMBER = 1001
nyct_feed_header = _descriptor.FieldDescriptor(
  name='nyct_feed_header', full_name='nyct_feed_header', index=0,
  number=1001, type=11, cpp_type=10, label=1,
  has_default_value=False, default_value=None,
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  serialized_options=None, file=DESCRIPTOR)
NYCT_TRIP_DESCRIPTOR_FIELD_NUMBER = 1001
nyct_trip_descriptor = _descriptor.FieldDescriptor(
  name='nyct_trip_descriptor', full_name='nyct_trip_descriptor', index=1,
  number=1001, type=11, cpp_type=10, label=1,
  has_default_value=False, default_value=None,
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  serialized_options=None, file=DESCRIPTOR)
NYCT_STOP_TIME_UPDATE_FIELD_NUMBER = 1001
nyct_stop_time_update = _descriptor.FieldDescriptor(
  name='nyct_stop_time_update', full_name='nyct_stop_time_update', index=2,
  number=1001, type=11, cpp_type=10, label=1,
  has_default_value=False, default_value=None,
  message_type=None, enum_type=None, containing_type=None,
  is_extension=True, extension_scope=None,
  serialized_options=None, file=DESCRIPTOR)

_NYCTTRIPDESCRIPTOR_DIRECTION = _descriptor.EnumDescriptor(
  name='Direction',
  full_name='NyctTripDescriptor.Direction',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='NORTH', index=0, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='EAST', index=1, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SOUTH', index=2, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='WEST', index=3, number=4,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=358,
  serialized_end=411,
)
_sym_db.RegisterEnumDescriptor(_NYCTTRIPDESCRIPTOR_DIRECTION)


_TRIPREPLACEMENTPERIOD = _descriptor.Descriptor(
  name='TripReplacementPeriod',
  full_name='TripReplacementPeriod',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='route_id', full_name='TripReplacementPeriod.route_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='replacement_period', full_name='TripReplacementPeriod.replacement_period', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=42,
  serialized_end=140,
)


_NYCTFEEDHEADER = _descriptor.Descriptor(
  name='NyctFeedHeader',
  full_name='NyctFeedHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='nyct_subway_version', full_name='NyctFeedHeader.nyct_subway_version', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='trip_replacement_period', full_name='NyctFeedHeader.trip_replacement_period', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=142,
  serialized_end=244,
)


_NYCTTRIPDESCRIPTOR = _descriptor.Descriptor(
  name='NyctTripDescriptor',
  full_name='NyctTripDescriptor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='train_id', full_name='NyctTripDescriptor.train_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='is_assigned', full_name='NyctTripDescriptor.is_assigned', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='direction', full_name='NyctTripDescriptor.direction', index=2,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _NYCTTRIPDESCRIPTOR_DIRECTION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=247,
  serialized_end=411,
)


_NYCTSTOPTIMEUPDATE = _descriptor.Descriptor(
  name='NyctStopTimeUpdate',
  full_name='NyctStopTimeUpdate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='scheduled_track', full_name='NyctStopTimeUpdate.scheduled_track', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='actual_track', full_name='NyctStopTimeUpdate.actual_track', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
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
  serialized_start=413,
  serialized_end=480,
)

_TRIPREPLACEMENTPERIOD.fields_by_name['replacement_period'].message_type = gtfs__realtime__pb2._TIMERANGE
_NYCTFEEDHEADER.fields_by_name['trip_replacement_period'].message_type = _TRIPREPLACEMENTPERIOD
_NYCTTRIPDESCRIPTOR.fields_by_name['direction'].enum_type = _NYCTTRIPDESCRIPTOR_DIRECTION
_NYCTTRIPDESCRIPTOR_DIRECTION.containing_type = _NYCTTRIPDESCRIPTOR
DESCRIPTOR.message_types_by_name['TripReplacementPeriod'] = _TRIPREPLACEMENTPERIOD
DESCRIPTOR.message_types_by_name['NyctFeedHeader'] = _NYCTFEEDHEADER
DESCRIPTOR.message_types_by_name['NyctTripDescriptor'] = _NYCTTRIPDESCRIPTOR
DESCRIPTOR.message_types_by_name['NyctStopTimeUpdate'] = _NYCTSTOPTIMEUPDATE
DESCRIPTOR.extensions_by_name['nyct_feed_header'] = nyct_feed_header
DESCRIPTOR.extensions_by_name['nyct_trip_descriptor'] = nyct_trip_descriptor
DESCRIPTOR.extensions_by_name['nyct_stop_time_update'] = nyct_stop_time_update
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TripReplacementPeriod = _reflection.GeneratedProtocolMessageType('TripReplacementPeriod', (_message.Message,), dict(
  DESCRIPTOR = _TRIPREPLACEMENTPERIOD,
  __module__ = 'nyct_subway_pb2'
  # @@protoc_insertion_point(class_scope:TripReplacementPeriod)
  ))
_sym_db.RegisterMessage(TripReplacementPeriod)

NyctFeedHeader = _reflection.GeneratedProtocolMessageType('NyctFeedHeader', (_message.Message,), dict(
  DESCRIPTOR = _NYCTFEEDHEADER,
  __module__ = 'nyct_subway_pb2'
  # @@protoc_insertion_point(class_scope:NyctFeedHeader)
  ))
_sym_db.RegisterMessage(NyctFeedHeader)

NyctTripDescriptor = _reflection.GeneratedProtocolMessageType('NyctTripDescriptor', (_message.Message,), dict(
  DESCRIPTOR = _NYCTTRIPDESCRIPTOR,
  __module__ = 'nyct_subway_pb2'
  # @@protoc_insertion_point(class_scope:NyctTripDescriptor)
  ))
_sym_db.RegisterMessage(NyctTripDescriptor)

NyctStopTimeUpdate = _reflection.GeneratedProtocolMessageType('NyctStopTimeUpdate', (_message.Message,), dict(
  DESCRIPTOR = _NYCTSTOPTIMEUPDATE,
  __module__ = 'nyct_subway_pb2'
  # @@protoc_insertion_point(class_scope:NyctStopTimeUpdate)
  ))
_sym_db.RegisterMessage(NyctStopTimeUpdate)

nyct_feed_header.message_type = _NYCTFEEDHEADER
gtfs__realtime__pb2.FeedHeader.RegisterExtension(nyct_feed_header)
nyct_trip_descriptor.message_type = _NYCTTRIPDESCRIPTOR
gtfs__realtime__pb2.TripDescriptor.RegisterExtension(nyct_trip_descriptor)
nyct_stop_time_update.message_type = _NYCTSTOPTIMEUPDATE
gtfs__realtime__pb2.TripUpdate.StopTimeUpdate.RegisterExtension(nyct_stop_time_update)

DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
