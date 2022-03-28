// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: scann/partitioning/kmeans_tree_partitioner.proto

#include "scann/partitioning/kmeans_tree_partitioner.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_scann_2ftrees_2fkmeans_5ftree_2fkmeans_5ftree_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_SerializedKMeansTree_scann_2ftrees_2fkmeans_5ftree_2fkmeans_5ftree_2eproto;
namespace research_scann {
class SerializedKMeansTreePartitionerDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<SerializedKMeansTreePartitioner> _instance;
} _SerializedKMeansTreePartitioner_default_instance_;
}  // namespace research_scann
static void InitDefaultsscc_info_SerializedKMeansTreePartitioner_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::research_scann::_SerializedKMeansTreePartitioner_default_instance_;
    new (ptr) ::research_scann::SerializedKMeansTreePartitioner();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::research_scann::SerializedKMeansTreePartitioner::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_SerializedKMeansTreePartitioner_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsscc_info_SerializedKMeansTreePartitioner_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto}, {
      &scc_info_SerializedKMeansTree_scann_2ftrees_2fkmeans_5ftree_2fkmeans_5ftree_2eproto.base,}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  PROTOBUF_FIELD_OFFSET(::research_scann::SerializedKMeansTreePartitioner, _has_bits_),
  PROTOBUF_FIELD_OFFSET(::research_scann::SerializedKMeansTreePartitioner, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::research_scann::SerializedKMeansTreePartitioner, kmeans_tree_),
  0,
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, 6, sizeof(::research_scann::SerializedKMeansTreePartitioner)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::research_scann::_SerializedKMeansTreePartitioner_default_instance_),
};

const char descriptor_table_protodef_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n0scann/partitioning/kmeans_tree_partiti"
  "oner.proto\022\016research_scann\032)scann/trees/"
  "kmeans_tree/kmeans_tree.proto\"\\\n\037Seriali"
  "zedKMeansTreePartitioner\0229\n\013kmeans_tree\030"
  "\001 \001(\0132$.research_scann.SerializedKMeansT"
  "ree"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto_deps[1] = {
  &::descriptor_table_scann_2ftrees_2fkmeans_5ftree_2fkmeans_5ftree_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto_sccs[1] = {
  &scc_info_SerializedKMeansTreePartitioner_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto_once;
static bool descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto = {
  &descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto_initialized, descriptor_table_protodef_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto, "scann/partitioning/kmeans_tree_partitioner.proto", 203,
  &descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto_once, descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto_sccs, descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto::offsets,
  file_level_metadata_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto, 1, file_level_enum_descriptors_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto, file_level_service_descriptors_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto), true);
namespace research_scann {

// ===================================================================

void SerializedKMeansTreePartitioner::InitAsDefaultInstance() {
  ::research_scann::_SerializedKMeansTreePartitioner_default_instance_._instance.get_mutable()->kmeans_tree_ = const_cast< ::research_scann::SerializedKMeansTree*>(
      ::research_scann::SerializedKMeansTree::internal_default_instance());
}
class SerializedKMeansTreePartitioner::_Internal {
 public:
  using HasBits = decltype(std::declval<SerializedKMeansTreePartitioner>()._has_bits_);
  static const ::research_scann::SerializedKMeansTree& kmeans_tree(const SerializedKMeansTreePartitioner* msg);
  static void set_has_kmeans_tree(HasBits* has_bits) {
    (*has_bits)[0] |= 1u;
  }
};

const ::research_scann::SerializedKMeansTree&
SerializedKMeansTreePartitioner::_Internal::kmeans_tree(const SerializedKMeansTreePartitioner* msg) {
  return *msg->kmeans_tree_;
}
void SerializedKMeansTreePartitioner::clear_kmeans_tree() {
  if (kmeans_tree_ != nullptr) kmeans_tree_->Clear();
  _has_bits_[0] &= ~0x00000001u;
}
SerializedKMeansTreePartitioner::SerializedKMeansTreePartitioner()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:research_scann.SerializedKMeansTreePartitioner)
}
SerializedKMeansTreePartitioner::SerializedKMeansTreePartitioner(const SerializedKMeansTreePartitioner& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      _has_bits_(from._has_bits_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  if (from.has_kmeans_tree()) {
    kmeans_tree_ = new ::research_scann::SerializedKMeansTree(*from.kmeans_tree_);
  } else {
    kmeans_tree_ = nullptr;
  }
  // @@protoc_insertion_point(copy_constructor:research_scann.SerializedKMeansTreePartitioner)
}

void SerializedKMeansTreePartitioner::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_SerializedKMeansTreePartitioner_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto.base);
  kmeans_tree_ = nullptr;
}

SerializedKMeansTreePartitioner::~SerializedKMeansTreePartitioner() {
  // @@protoc_insertion_point(destructor:research_scann.SerializedKMeansTreePartitioner)
  SharedDtor();
}

void SerializedKMeansTreePartitioner::SharedDtor() {
  if (this != internal_default_instance()) delete kmeans_tree_;
}

void SerializedKMeansTreePartitioner::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const SerializedKMeansTreePartitioner& SerializedKMeansTreePartitioner::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_SerializedKMeansTreePartitioner_scann_2fpartitioning_2fkmeans_5ftree_5fpartitioner_2eproto.base);
  return *internal_default_instance();
}


void SerializedKMeansTreePartitioner::Clear() {
// @@protoc_insertion_point(message_clear_start:research_scann.SerializedKMeansTreePartitioner)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    GOOGLE_DCHECK(kmeans_tree_ != nullptr);
    kmeans_tree_->Clear();
  }
  _has_bits_.Clear();
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* SerializedKMeansTreePartitioner::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  _Internal::HasBits has_bits{};
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // optional .research_scann.SerializedKMeansTree kmeans_tree = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ctx->ParseMessage(mutable_kmeans_tree(), ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  _has_bits_.Or(has_bits);
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool SerializedKMeansTreePartitioner::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:research_scann.SerializedKMeansTreePartitioner)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional .research_scann.SerializedKMeansTree kmeans_tree = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (10 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
               input, mutable_kmeans_tree()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:research_scann.SerializedKMeansTreePartitioner)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:research_scann.SerializedKMeansTreePartitioner)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void SerializedKMeansTreePartitioner::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:research_scann.SerializedKMeansTreePartitioner)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .research_scann.SerializedKMeansTree kmeans_tree = 1;
  if (cached_has_bits & 0x00000001u) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      1, _Internal::kmeans_tree(this), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:research_scann.SerializedKMeansTreePartitioner)
}

::PROTOBUF_NAMESPACE_ID::uint8* SerializedKMeansTreePartitioner::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:research_scann.SerializedKMeansTreePartitioner)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  cached_has_bits = _has_bits_[0];
  // optional .research_scann.SerializedKMeansTree kmeans_tree = 1;
  if (cached_has_bits & 0x00000001u) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, _Internal::kmeans_tree(this), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:research_scann.SerializedKMeansTreePartitioner)
  return target;
}

size_t SerializedKMeansTreePartitioner::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:research_scann.SerializedKMeansTreePartitioner)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // optional .research_scann.SerializedKMeansTree kmeans_tree = 1;
  cached_has_bits = _has_bits_[0];
  if (cached_has_bits & 0x00000001u) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
        *kmeans_tree_);
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void SerializedKMeansTreePartitioner::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:research_scann.SerializedKMeansTreePartitioner)
  GOOGLE_DCHECK_NE(&from, this);
  const SerializedKMeansTreePartitioner* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<SerializedKMeansTreePartitioner>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:research_scann.SerializedKMeansTreePartitioner)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:research_scann.SerializedKMeansTreePartitioner)
    MergeFrom(*source);
  }
}

void SerializedKMeansTreePartitioner::MergeFrom(const SerializedKMeansTreePartitioner& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:research_scann.SerializedKMeansTreePartitioner)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.has_kmeans_tree()) {
    mutable_kmeans_tree()->::research_scann::SerializedKMeansTree::MergeFrom(from.kmeans_tree());
  }
}

void SerializedKMeansTreePartitioner::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:research_scann.SerializedKMeansTreePartitioner)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SerializedKMeansTreePartitioner::CopyFrom(const SerializedKMeansTreePartitioner& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:research_scann.SerializedKMeansTreePartitioner)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SerializedKMeansTreePartitioner::IsInitialized() const {
  return true;
}

void SerializedKMeansTreePartitioner::InternalSwap(SerializedKMeansTreePartitioner* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_has_bits_[0], other->_has_bits_[0]);
  swap(kmeans_tree_, other->kmeans_tree_);
}

::PROTOBUF_NAMESPACE_ID::Metadata SerializedKMeansTreePartitioner::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace research_scann
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::research_scann::SerializedKMeansTreePartitioner* Arena::CreateMaybeMessage< ::research_scann::SerializedKMeansTreePartitioner >(Arena* arena) {
  return Arena::CreateInternal< ::research_scann::SerializedKMeansTreePartitioner >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
