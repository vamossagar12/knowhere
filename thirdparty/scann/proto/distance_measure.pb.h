// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: scann/proto/distance_measure.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_scann_2fproto_2fdistance_5fmeasure_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_scann_2fproto_2fdistance_5fmeasure_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3009000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3009002 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_scann_2fproto_2fdistance_5fmeasure_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_scann_2fproto_2fdistance_5fmeasure_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_scann_2fproto_2fdistance_5fmeasure_2eproto;
namespace research_scann {
class DistanceMeasureConfig;
class DistanceMeasureConfigDefaultTypeInternal;
extern DistanceMeasureConfigDefaultTypeInternal _DistanceMeasureConfig_default_instance_;
class DistanceMeasureParamsConfig;
class DistanceMeasureParamsConfigDefaultTypeInternal;
extern DistanceMeasureParamsConfigDefaultTypeInternal _DistanceMeasureParamsConfig_default_instance_;
}  // namespace research_scann
PROTOBUF_NAMESPACE_OPEN
template<> ::research_scann::DistanceMeasureConfig* Arena::CreateMaybeMessage<::research_scann::DistanceMeasureConfig>(Arena*);
template<> ::research_scann::DistanceMeasureParamsConfig* Arena::CreateMaybeMessage<::research_scann::DistanceMeasureParamsConfig>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace research_scann {

// ===================================================================

class DistanceMeasureConfig :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:research_scann.DistanceMeasureConfig) */ {
 public:
  DistanceMeasureConfig();
  virtual ~DistanceMeasureConfig();

  DistanceMeasureConfig(const DistanceMeasureConfig& from);
  DistanceMeasureConfig(DistanceMeasureConfig&& from) noexcept
    : DistanceMeasureConfig() {
    *this = ::std::move(from);
  }

  inline DistanceMeasureConfig& operator=(const DistanceMeasureConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline DistanceMeasureConfig& operator=(DistanceMeasureConfig&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const DistanceMeasureConfig& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const DistanceMeasureConfig* internal_default_instance() {
    return reinterpret_cast<const DistanceMeasureConfig*>(
               &_DistanceMeasureConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(DistanceMeasureConfig& a, DistanceMeasureConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(DistanceMeasureConfig* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline DistanceMeasureConfig* New() const final {
    return CreateMaybeMessage<DistanceMeasureConfig>(nullptr);
  }

  DistanceMeasureConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<DistanceMeasureConfig>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const DistanceMeasureConfig& from);
  void MergeFrom(const DistanceMeasureConfig& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  #else
  bool MergePartialFromCodedStream(
      ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(DistanceMeasureConfig* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "research_scann.DistanceMeasureConfig";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_scann_2fproto_2fdistance_5fmeasure_2eproto);
    return ::descriptor_table_scann_2fproto_2fdistance_5fmeasure_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kDistanceMeasureFieldNumber = 1,
    kDistanceMeasureParamsFieldNumber = 2,
  };
  // optional string distance_measure = 1 [default = "SquaredL2Distance"];
  bool has_distance_measure() const;
  void clear_distance_measure();
  const std::string& distance_measure() const;
  void set_distance_measure(const std::string& value);
  void set_distance_measure(std::string&& value);
  void set_distance_measure(const char* value);
  void set_distance_measure(const char* value, size_t size);
  std::string* mutable_distance_measure();
  std::string* release_distance_measure();
  void set_allocated_distance_measure(std::string* distance_measure);

  // optional .research_scann.DistanceMeasureParamsConfig distance_measure_params = 2;
  bool has_distance_measure_params() const;
  void clear_distance_measure_params();
  const ::research_scann::DistanceMeasureParamsConfig& distance_measure_params() const;
  ::research_scann::DistanceMeasureParamsConfig* release_distance_measure_params();
  ::research_scann::DistanceMeasureParamsConfig* mutable_distance_measure_params();
  void set_allocated_distance_measure_params(::research_scann::DistanceMeasureParamsConfig* distance_measure_params);

  // @@protoc_insertion_point(class_scope:research_scann.DistanceMeasureConfig)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  public:
  static ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<std::string> _i_give_permission_to_break_this_code_default_distance_measure_;
  private:
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr distance_measure_;
  ::research_scann::DistanceMeasureParamsConfig* distance_measure_params_;
  friend struct ::TableStruct_scann_2fproto_2fdistance_5fmeasure_2eproto;
};
// -------------------------------------------------------------------

class DistanceMeasureParamsConfig :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:research_scann.DistanceMeasureParamsConfig) */ {
 public:
  DistanceMeasureParamsConfig();
  virtual ~DistanceMeasureParamsConfig();

  DistanceMeasureParamsConfig(const DistanceMeasureParamsConfig& from);
  DistanceMeasureParamsConfig(DistanceMeasureParamsConfig&& from) noexcept
    : DistanceMeasureParamsConfig() {
    *this = ::std::move(from);
  }

  inline DistanceMeasureParamsConfig& operator=(const DistanceMeasureParamsConfig& from) {
    CopyFrom(from);
    return *this;
  }
  inline DistanceMeasureParamsConfig& operator=(DistanceMeasureParamsConfig&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  inline const ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::PROTOBUF_NAMESPACE_ID::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const DistanceMeasureParamsConfig& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const DistanceMeasureParamsConfig* internal_default_instance() {
    return reinterpret_cast<const DistanceMeasureParamsConfig*>(
               &_DistanceMeasureParamsConfig_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(DistanceMeasureParamsConfig& a, DistanceMeasureParamsConfig& b) {
    a.Swap(&b);
  }
  inline void Swap(DistanceMeasureParamsConfig* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline DistanceMeasureParamsConfig* New() const final {
    return CreateMaybeMessage<DistanceMeasureParamsConfig>(nullptr);
  }

  DistanceMeasureParamsConfig* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<DistanceMeasureParamsConfig>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const DistanceMeasureParamsConfig& from);
  void MergeFrom(const DistanceMeasureParamsConfig& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  #else
  bool MergePartialFromCodedStream(
      ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(DistanceMeasureParamsConfig* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "research_scann.DistanceMeasureParamsConfig";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_scann_2fproto_2fdistance_5fmeasure_2eproto);
    return ::descriptor_table_scann_2fproto_2fdistance_5fmeasure_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kFilenameFieldNumber = 1,
  };
  // optional string filename = 1;
  bool has_filename() const;
  void clear_filename();
  const std::string& filename() const;
  void set_filename(const std::string& value);
  void set_filename(std::string&& value);
  void set_filename(const char* value);
  void set_filename(const char* value, size_t size);
  std::string* mutable_filename();
  std::string* release_filename();
  void set_allocated_filename(std::string* filename);

  // @@protoc_insertion_point(class_scope:research_scann.DistanceMeasureParamsConfig)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::HasBits<1> _has_bits_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr filename_;
  friend struct ::TableStruct_scann_2fproto_2fdistance_5fmeasure_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// DistanceMeasureConfig

// optional string distance_measure = 1 [default = "SquaredL2Distance"];
inline bool DistanceMeasureConfig::has_distance_measure() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void DistanceMeasureConfig::clear_distance_measure() {
  distance_measure_.ClearToDefaultNoArena(&::research_scann::DistanceMeasureConfig::_i_give_permission_to_break_this_code_default_distance_measure_.get());
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& DistanceMeasureConfig::distance_measure() const {
  // @@protoc_insertion_point(field_get:research_scann.DistanceMeasureConfig.distance_measure)
  return distance_measure_.GetNoArena();
}
inline void DistanceMeasureConfig::set_distance_measure(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  distance_measure_.SetNoArena(&::research_scann::DistanceMeasureConfig::_i_give_permission_to_break_this_code_default_distance_measure_.get(), value);
  // @@protoc_insertion_point(field_set:research_scann.DistanceMeasureConfig.distance_measure)
}
inline void DistanceMeasureConfig::set_distance_measure(std::string&& value) {
  _has_bits_[0] |= 0x00000001u;
  distance_measure_.SetNoArena(
    &::research_scann::DistanceMeasureConfig::_i_give_permission_to_break_this_code_default_distance_measure_.get(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:research_scann.DistanceMeasureConfig.distance_measure)
}
inline void DistanceMeasureConfig::set_distance_measure(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000001u;
  distance_measure_.SetNoArena(&::research_scann::DistanceMeasureConfig::_i_give_permission_to_break_this_code_default_distance_measure_.get(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:research_scann.DistanceMeasureConfig.distance_measure)
}
inline void DistanceMeasureConfig::set_distance_measure(const char* value, size_t size) {
  _has_bits_[0] |= 0x00000001u;
  distance_measure_.SetNoArena(&::research_scann::DistanceMeasureConfig::_i_give_permission_to_break_this_code_default_distance_measure_.get(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:research_scann.DistanceMeasureConfig.distance_measure)
}
inline std::string* DistanceMeasureConfig::mutable_distance_measure() {
  _has_bits_[0] |= 0x00000001u;
  // @@protoc_insertion_point(field_mutable:research_scann.DistanceMeasureConfig.distance_measure)
  return distance_measure_.MutableNoArena(&::research_scann::DistanceMeasureConfig::_i_give_permission_to_break_this_code_default_distance_measure_.get());
}
inline std::string* DistanceMeasureConfig::release_distance_measure() {
  // @@protoc_insertion_point(field_release:research_scann.DistanceMeasureConfig.distance_measure)
  if (!has_distance_measure()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return distance_measure_.ReleaseNonDefaultNoArena(&::research_scann::DistanceMeasureConfig::_i_give_permission_to_break_this_code_default_distance_measure_.get());
}
inline void DistanceMeasureConfig::set_allocated_distance_measure(std::string* distance_measure) {
  if (distance_measure != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  distance_measure_.SetAllocatedNoArena(&::research_scann::DistanceMeasureConfig::_i_give_permission_to_break_this_code_default_distance_measure_.get(), distance_measure);
  // @@protoc_insertion_point(field_set_allocated:research_scann.DistanceMeasureConfig.distance_measure)
}

// optional .research_scann.DistanceMeasureParamsConfig distance_measure_params = 2;
inline bool DistanceMeasureConfig::has_distance_measure_params() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void DistanceMeasureConfig::clear_distance_measure_params() {
  if (distance_measure_params_ != nullptr) distance_measure_params_->Clear();
  _has_bits_[0] &= ~0x00000002u;
}
inline const ::research_scann::DistanceMeasureParamsConfig& DistanceMeasureConfig::distance_measure_params() const {
  const ::research_scann::DistanceMeasureParamsConfig* p = distance_measure_params_;
  // @@protoc_insertion_point(field_get:research_scann.DistanceMeasureConfig.distance_measure_params)
  return p != nullptr ? *p : *reinterpret_cast<const ::research_scann::DistanceMeasureParamsConfig*>(
      &::research_scann::_DistanceMeasureParamsConfig_default_instance_);
}
inline ::research_scann::DistanceMeasureParamsConfig* DistanceMeasureConfig::release_distance_measure_params() {
  // @@protoc_insertion_point(field_release:research_scann.DistanceMeasureConfig.distance_measure_params)
  _has_bits_[0] &= ~0x00000002u;
  ::research_scann::DistanceMeasureParamsConfig* temp = distance_measure_params_;
  distance_measure_params_ = nullptr;
  return temp;
}
inline ::research_scann::DistanceMeasureParamsConfig* DistanceMeasureConfig::mutable_distance_measure_params() {
  _has_bits_[0] |= 0x00000002u;
  if (distance_measure_params_ == nullptr) {
    auto* p = CreateMaybeMessage<::research_scann::DistanceMeasureParamsConfig>(GetArenaNoVirtual());
    distance_measure_params_ = p;
  }
  // @@protoc_insertion_point(field_mutable:research_scann.DistanceMeasureConfig.distance_measure_params)
  return distance_measure_params_;
}
inline void DistanceMeasureConfig::set_allocated_distance_measure_params(::research_scann::DistanceMeasureParamsConfig* distance_measure_params) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete distance_measure_params_;
  }
  if (distance_measure_params) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      distance_measure_params = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, distance_measure_params, submessage_arena);
    }
    _has_bits_[0] |= 0x00000002u;
  } else {
    _has_bits_[0] &= ~0x00000002u;
  }
  distance_measure_params_ = distance_measure_params;
  // @@protoc_insertion_point(field_set_allocated:research_scann.DistanceMeasureConfig.distance_measure_params)
}

// -------------------------------------------------------------------

// DistanceMeasureParamsConfig

// optional string filename = 1;
inline bool DistanceMeasureParamsConfig::has_filename() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void DistanceMeasureParamsConfig::clear_filename() {
  filename_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  _has_bits_[0] &= ~0x00000001u;
}
inline const std::string& DistanceMeasureParamsConfig::filename() const {
  // @@protoc_insertion_point(field_get:research_scann.DistanceMeasureParamsConfig.filename)
  return filename_.GetNoArena();
}
inline void DistanceMeasureParamsConfig::set_filename(const std::string& value) {
  _has_bits_[0] |= 0x00000001u;
  filename_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:research_scann.DistanceMeasureParamsConfig.filename)
}
inline void DistanceMeasureParamsConfig::set_filename(std::string&& value) {
  _has_bits_[0] |= 0x00000001u;
  filename_.SetNoArena(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:research_scann.DistanceMeasureParamsConfig.filename)
}
inline void DistanceMeasureParamsConfig::set_filename(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  _has_bits_[0] |= 0x00000001u;
  filename_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:research_scann.DistanceMeasureParamsConfig.filename)
}
inline void DistanceMeasureParamsConfig::set_filename(const char* value, size_t size) {
  _has_bits_[0] |= 0x00000001u;
  filename_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:research_scann.DistanceMeasureParamsConfig.filename)
}
inline std::string* DistanceMeasureParamsConfig::mutable_filename() {
  _has_bits_[0] |= 0x00000001u;
  // @@protoc_insertion_point(field_mutable:research_scann.DistanceMeasureParamsConfig.filename)
  return filename_.MutableNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline std::string* DistanceMeasureParamsConfig::release_filename() {
  // @@protoc_insertion_point(field_release:research_scann.DistanceMeasureParamsConfig.filename)
  if (!has_filename()) {
    return nullptr;
  }
  _has_bits_[0] &= ~0x00000001u;
  return filename_.ReleaseNonDefaultNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline void DistanceMeasureParamsConfig::set_allocated_filename(std::string* filename) {
  if (filename != nullptr) {
    _has_bits_[0] |= 0x00000001u;
  } else {
    _has_bits_[0] &= ~0x00000001u;
  }
  filename_.SetAllocatedNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), filename);
  // @@protoc_insertion_point(field_set_allocated:research_scann.DistanceMeasureParamsConfig.filename)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace research_scann

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_scann_2fproto_2fdistance_5fmeasure_2eproto
