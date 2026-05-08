#ifndef __GENERATE_QNN_UTILS_HPP__
#define __GENERATE_QNN_UTILS_HPP__

#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include "android_memory_allocator.h"
#include <model.h>
#include <tokenizers_cpp.h>

using ModelHandle = std::unique_ptr<ml::train::Model>;
using IO_TensorType = ml::train::TensorDim::IO_TensorType;

extern std::mt19937 rng;
extern std::chrono::duration<double> raw_exec_seconds;

std::tuple<uint16_t *, uint16_t *> get_cos_sin (int context_size, int pos_dim,
    const double theta, const std::string &rope_type = "default",
    double partial_rotary_factor = 1.0, double rope_scaling_factor = 1.0);

void process_key (uint8_t *pointer, int row, int column, uint8_t *dest, int idx,
    int dest_row_length, int src_row_length);

void process_value (uint8_t *pointer, int row, int column, uint8_t *dest, int idx);

void fill_attention_mask_with_length (
    int rows, int columns, int length, uint16_t *attention_mask);

void fill_attention_mask_with_prev_length (
    int rows, int columns, int length, uint16_t *attention_mask);

uint16_t *get_zero_memory (int size, int zero_point);

int sample (uint16_t *pointer, int length, int *tokens, int number_of_tokens,
    float logit_scale, int logit_offset, float repetition_penalty,
    float temperature, float top_p, int top_k);

#endif // __GENERATE_QNN_UTILS_HPP__
