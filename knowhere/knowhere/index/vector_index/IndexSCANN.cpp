// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include "knowhere/index/vector_index/IndexSCANN.h"

#include <format>
#include <omp.h>
#include <sstream>
#include <string>
#include "scann/utils/io_npy.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "knowhere/index/vector_index/helpers/IndexParameter.h"
#include "scann/base/single_machine_factory_options.h"

namespace milvus {
namespace knowhere {


BinarySet
IndexSCANN::Serialize(const Config& config) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialize or trained");
    }
    
    TF_ASSIGN_OR_RETURN(auto opts, index_->ExtractSingleMachineFactoryOptions());
    BinarySet res_set;
    res_set.Append("scann_config", index_->GetBinary(index_->config()), index_->config()->ByteSizeLong());
    
    /*
    std::shared_ptr<uint8_t[]> single_machine_factory_options(new uint8_t[sizeof(opts)]);
    memcpy(single_machine_factory_options.get(), single_machine_factory_options, sizeof(opts));
    res_set.Append("single_machine_factory_options", single_machine_factory_options, sizeof(opts));
    */
    
    if (opts.ah_codebook != nullptr) 
        res_set.Append("ah_codebook", index->GetBinary(opts.ah_codebook.get()), index_->GetSize(opts.ah_codebook.get()));
    if (opts.serialized_partitioner != nullptr)
        res_set.Append("serialized_partitioner", index->GetBinary(opts.serialized_partitioner.get()), index_->GetSize(opts.serialized_partitioner.get()));
    
    DatapointIndex n_points = kInvalidDatapointIndex;

    // datapoints_to_token
    if (opts.datapoints_by_token != nullptr) {
        vector<int32_t> datapoint_to_token(index_->n_points());
        for (const auto& [token_idx, dps] : Enumerate(*opts.datapoints_by_token))
            for (auto dp_idx : dps) datapoint_to_token[dp_idx] = token_idx;
        auto datapoints_size = datapoint_to_token.size() * sizeof(int32_t);

        //std::vector<Binary> datapoints_to_token_binary = VectorToNumpy(datapoint_to_token);
        std::shared_ptr<uint8_t[]> datapoints(new uint8_t[datapoints_size]);
        memcpy(datapoints.get(), datapoint_to_token, datapoints_size);
        res_set.Append("datapoint_to_token", datapoints, datapoints_size);
        n_points = index_->n_points();
    }
    // hashed_dataset
    if (opts.hashed_dataset != nullptr) {
        DenseDataset<uint8_t>* hashed_dataset = opts.hashed_dataset();
        ConstSpan<uint8_t> hashed_dataset_const_span = DatasetToSpan(hashed_dataset);
        size_t hashed_dataset_size = sizeof(hashed_dataset_const_span) * sizeof(uint8_t);
        std::shared_ptr<uint8_t[]> hashed_data_points(new uint8_t[hashed_data_size]);
        memcpy(hashed_data_points.get(), hashed_dataset_const_span.data(), hashed_dataset_size);
        res_set.Append("hashed_dataset", hashed_data_points, hashed_data_size);
        n_points = hashed_dataset->size();
    }
    if (opts.pre_quantized_fixed_point != nullptr) {
        auto fixed_point = opts.pre_quantized_fixed_point;
        // int8_dataset
        if (fixed_point->fixed_point_dataset != nullptr) {
            DenseDataset<int8_t>* fixed_point_dataset = fixed_point->fixed_point_dataset.get();
            ConstSpan<uint8_t> fixed_point_dataset_const_span = DatasetToSpan(fixed_point_dataset);
            size_t fixed_point_data_size = sizeof(fixed_point_dataset) * sizeof(uint8_t);
            std::shared_ptr<uint8_t[]> fixed_point_data_points(new uint8_t[fixed_point_data_size]);
            memcpy(fixed_point_data_points.get(), fixed_point_dataset_const_span, fixed_point_data_size);
            res_set.Append("int8_dataset", fixed_point_data_points, fixed_point_data_size);
            n_points = fixed_point_dataset->size();
        }
        // int8_multipliers
        if (fixed_point->multiplier_by_dimension != nullptr) {
            vector<float>* multiplier_by_dimension = fixed_point->multiplier_by_dimension.get();
            auto multipliers_size = multiplier_by_dimension->size() * sizeof(float);
            std::shared_ptr<uint8_t[]> int8_multipliers(new uint8_t[multipliers_size]);
            memcpy(int8_multipliers.get(), *multiplier_by_dimension, multipliers_size);
            res_set.Append("int8_multipliers", int8_multipliers, multipliers_size);
        }
        //dp_norms
        if (fixed_point->squared_l2_norm_by_datapoint != nullptr) {
            vector<float>* norms = fixed_point->squared_l2_norm_by_datapoint.get();
            auto norms_size = norms->size() * sizeof(float);
            std::shared_ptr<uint8_t[]> dp_norms(new uint8_t[norms_size]);
            memcpy(dp_norms.get(), *norms, norms_size);
            res_set.Append("dp_norms", dp_norms, norms_size);
        }
    }
    
    if (index_->needs_dataset()) {
        if (scann_->dataset() == nullptr) 
            KNOWHERE_THROW_MSG("Searcher needs original dataset but none is present.");
        
        // dataset
        auto dataset = dynamic_cast<const DenseDataset<float>*>(scann_->dataset());
        if (dataset == nullptr)
            KNOWHERE_THROW_MSG("Failed to cast dataset to DenseDataset<float>.");
        DenseDataset<float>* dense_dataset = (*dataset)->data();
        ConstSpan<float> dense_dataset_const_span = DatasetToSpan(dense_dataset);
        auto dense_dataset_size = dense_dataset->size() * sizeof(float);
        std::shared_ptr<uint8_t[]> dataset(new uint8_t[dense_dataset_size]);
        memcpy(dataset.get(), dense_dataset_const_span, dense_dataset_size);
        res_set.Append("dataset", dataset, dense_dataset_size);
    }
    res_set.Append("n_points", n_points, sizeof(n_points));
   return res_set;
}


void Load(const BinarySet& index_binary) {
    Assemble(const_cast<BinarySet&>(index_binary));
    auto scann_config_binary = index_binary.GetByName("scann_config");
    auto retrieved_config = GetProto(scann_config_binary->data->get(), scann_config_binary->size);
    ScannConfig config;
    config.CopyFrom(*(retrieved_config));

    // TODO explore ah_codebook and serialized_partitioner
    //auto scann_config_binary = index_binary.GetByName("scann_config");
    ConstSpan<int32_t> tokenization;
    if (index_binary.Contains("datapoint_to_token")) {
        auto datapoint_to_token_binary = index_binary.GetByName("datapoint_to_token");
        vector<int32_t> datapoint_to_token;
        datapoint_to_token.resize(datapoint_to_token_binary->size / sizeof(int32_t));
        memcpy((void*)datapoint_to_token.data(), datapoint_to_token_binary->data.get, datapoint_to_token_binary->size);
        tokenization = ConstSpan<int32_t>(datapoint_to_token.data(), datapoint_to_token.size());
    }

    ConstSpan<uint8_t> hashed_span;
    if (index_binary.Contains("hashed_dataset")) {
        auto hashed_dataset = index_binary.GetByName("hashed_dataset");
        uint8_t* arr = new uint8_t[hashed_dataset->size/sizeof(uint8_t)];
        memcpy((void*)arr, hashed_dataset.data.get(), hashed_dataset->size);
        hashed_span = ConstSpan<uint8_t>(arr, arr.size());
    }

    ConstSpan<int8_t> int8_span;
    if (index_binary.Contains("int8_dataset")) {
        auto int8_dataset = index_binary.GetByName("int8_dataset");
        int8_t* arr = new int8_t[int8_dataset->size/sizeof(int8_t)];
        memcpy((void*)arr, int8_dataset.data.get(), int8_dataset->size);
        int8_span = ConstSpan<int8_t>(arr, arr.size());
    }

    ConstSpan<float> mult_span, norm_span;
    if (index_binary.Contains("int8_multipliers")) {
        auto int8_multipliers = index_binary.GetByName("int8_multipliers");
        float* arr = new int8_t[int8_multipliers->size/sizeof(float)];
        memcpy((void*)arr, int8_multipliers.data.get(), int8_multipliers->size);
        mult_span = ConstSpan<float>(arr, arr.size());
    }

    if (index_binary.Contains("dp_norms")) {
        auto dp_norms = index_binary.GetByName("dp_norms");
        float* arr = new int8_t[dp_norms->size/sizeof(float)];
        memcpy((void*)arr, dp_norms.data.get(), dp_norms->size);
        norm_span = ConstSpan<float>(arr, arr.size());
    }

    ConstSpan<float> dataset;
    if (index_binary.Contains("dataset")) {
        auto dataset_binary = index_binary.GetByName("dataset");
        float* arr = new int8_t[dataset_binary->size/sizeof(float)];
        memcpy((void*)arr, dataset_binary.data.get(), dataset_binary->size);
        dataset = ConstSpan<float>(arr, arr.size());
    }

    auto n_points = index_binary.GetByName("n_points");
    uint32_t datapoint_index;
    memcpy(&datapoint_index, n_points->data.get(), static_cast<size_t>(n_points->size));

    SingleMachineFactoryOptions opts;
    if (index_binary.Contains("ah_codebook")) {
        auto ah_codebook_binary = index_binary.GetByName("ah_codebook");
        auto ah_codebook = GetProto(ah_codebook_binary->data->get(), ah_codebook_binary->size);
        opts.ah_codebook = std::make_shared<CentersForAllSubspaces>();
        opts.ah_codebook.get().CopyFrom(*(ah_codebook));
    }

    if (index_binary.Contains("serialized_partitioner")) {
        auto serialized_partitioner_binary = index_binary.GetByName("serialized_partitioner");
        auto serialized_partitioner = GetProto(serialized_partitioner_binary->data->get(), serialized_partitioner_binary->size);
        opts.serialized_partitioner = std::make_shared<SerializedPartitioner>();
        opts.serialized_partitioner.get().CopyFrom(*(serialized_partitioner));
    }
    
    index_ = std::shared_ptr<research_scann::ScannInterface>(research_scann::ScannInterface::Initialize(config, opts, dataset, tokenization, hashed_span, int8_span, mult_span, norm_span, datapoint_index));
}

void
IndexSCANN::BuildAll(const DatasetPtr& dataset_ptr, const Config& config) {
    if (index_) {
        // it is built already
        LOG_KNOWHERE_DEBUG_ << "IndexSCANN::BuildAll: index_ has been built!";
        return;
    }

    GET_TENSOR_DATA_DIM(dataset_ptr)

    std::string distance_measure = "DotProductDistance";
    metric_type_ = config[Metric::TYPE];
    bool spherical = true;
    if (metric_type_ == Metric::L2) {
        spherical = false;
        distance_measure = "SquaredL2Distance";
    }
    size_t num_leaves = config[IndexParams::num_leaves].get<int64_t>();
    size_t num_leaves_to_search = config[IndexParams::num_leaves_to_search].get<int64_t>();
    auto k = config[meta::TOPK].get<int64_t>();
    const int dimensions_per_block = 2; // It's always set to 2 .
    GET_TENSOR_DATA_DIM(dataset_ptr)
    std::string scann_config = "";
    std::string scoring = config[IndexParams::scoring].get<std::string>();
    // Need to set 
    scann_config = R("
                    num_neighbors: " k
                    "distance_measure: "distance_measure
                    \"");
    if (scoring.compare("brute_force")) {
        scann_config += R("brute_force {
                            fixed_point {
                                enabled: false
                            }
                        }");
    } else {
        float anisotropic_quantization_threshold = config[IndexParams::anisotropic_quantization_threshold].get<float>();
        if (dim % dimensions_per_block == 0) {
            std::string proj_config = R("
                                        projection_type: CHUNK
                                        num_blocks: "(dim / dimensions_per_block)
                                        "num_dims_per_block: dimensions_per_block
                                    ");
        } else {
            std::string proj_config = R("
                                        projection_type: VARIABLE_CHUNK
                                        variable_blocks {
                                            num_blocks: "(dim / dimensions_per_block)
                                            "num_dims_per_block: dimensions_per_block
                                        }
                                        variable_blocks {
                                            num_blocks: 1
                                            num_dims_per_block: "(dim % dimensions_per_block)
                                        "}");
        }
        // For some of the config values, using default values.
        scann_config += R("hash {
                            asymmetric_hash {
                                lookup_type: INT8_LUT16
                                use_residual_quantization: None
                                use_global_topn: false
                                quantization_distance {
                                    distance_measure: SquaredL2Distance
                                }
                                num_clusters_per_block: 16
                                projection {
                                    input_dim: " + dim
                                    proj_config
                                "}
                                noise_shaping_threshold:" + anisotropic_quantization_threshold
                                "expected_sample_size: 10000
                                min_cluster_size: 100
                                max_clustering_iterations: 10
                                }
                            }
                        }");
    }
    // Configuring Partitioning during training.
    if (num_leaves_to_search != -1) {
        std::string partition_type = spherical? "SPHERICAL": "GENERIC";
        scann_config += R("
                            partitioning {
                            num_children: " k
                            "min_cluster_size: 50
                            max_clustering_iterations: 12
                            single_machine_center_initialization: {
                                RANDOM_INITIALIZATION
                            }
                            partitioning_distance {
                                distance_measure: SquaredL2Distance
                            }
                            query_spilling {
                                spilling_type: FIXED_NUMBER_OF_CENTERS
                                max_spill_centers: "num_leaves_to_search
                            "}
                            expected_sample_size: 100000
                            query_tokenization_distance_override: "distance_measure
                            "partitioning_type: {
                                "partition_type
                            "}
                            query_tokenization_type: {
                                FIXED_POINT_INT8}
                        }
                    ");
    }

    size_t reordering_num_neighbors = config[IndexParams::reordering_num_neighbors].get<int64_t>();
    // Reordering
    if (reordering_num_neighbors != -1) {
        scann_config += R("
                            exact_reordering {
                                approx_num_neighbors: "reordering_num_neighbors
                                "fixed_point {
                                    enabled: False
                                }
                            }");
    }

    ConstSpan<float> dataset(reinterpret_cast<const float*>(p_data), rows);
    ScannInterface scann_;
    RuntimeErrorIfNotOk( "Error initializing searcher: ", scann_.Initialize(dataset, dim, *config, 0));
    index_ = std::make_shared<research_scann::ScannInterface>(scann_); 
}

DatasetPtr
IndexSCANN::Query(const DatasetPtr& dataset_ptr, const Config& config, const faiss::BitsetView bitset) {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialized or trained");
    }
    GET_TENSOR_DATA_DIM(dataset_ptr)

    size_t k = config[meta::TOPK].get<int64_t>();
    auto pre_reorder_nn = config[meta::PRE_REORDER_NN].get<int64_t>();
    auto leaves = config[meta::LEAVES_TO_SEARCH].get<int64_t>();
    auto p_id = static_cast<int64_t*>(malloc(id_size * rows));
    auto p_dist = static_cast<float*>(malloc(dist_size * rows));

    int leaves = config[IndexParams::num_leaves].get<int64_t>());

// TODO: I think we can even use SearchBatched function from SCANN.
#pragma omp parallel for
    for (unsigned int i = 0; i < rows; ++i) {
        DatapointPtr<float> ptr(nullptr, static_cast<const float*>(p_data) + i * dim, rows, rows);
        NNResultsVector res;
        auto status = index_->Search(ptr, &res, final_nn, pre_reorder_nn, leaves);
        size_t result_num = result.size();
        // TODO: there should be a way to do this without the pybind11 arrays. Doing it this way for now.
        pybind11::array_t<DatapointIndex> indices(res.size());
        pybind11::array_t<float> distances(res.size());
        auto idx_ptr = reinterpret_cast<DatapointIndex*>(indices.request().ptr);
        auto dis_ptr = reinterpret_cast<float*>(distances.request().ptr);
        index_->ReshapeNNResult(res, idx_ptr, dis_ptr);
        auto local_p_id = p_id + k * i;
        auto local_p_dist = p_dist + k * i;
        memcpy(local_p_id, indices.data(), result_num * sizeof(int64_t));
        memcpy(local_p_dist, distances.data(), result_num * sizeof(float));
    }

    auto ret_ds = std::make_shared<Dataset>();
    ret_ds->Set(meta::IDS, p_id);
    ret_ds->Set(meta::DISTANCE, p_dist);
    return ret_ds;
}

int64_t
IndexSCANN::Count() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialized");
    }
    return index_->n_points();
}

int64_t Dim() {
    if (!index_) {
        KNOWHERE_THROW_MSG("index not initialized");
    }
    return index_->dimensionality();
}


}  // namespace knowhere
}  // namespace milvus
