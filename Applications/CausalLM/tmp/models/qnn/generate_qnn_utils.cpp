#include "generate_qnn_utils.h"
#include "android_memory_allocator.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <random>
#include <utility>
#include <vector>

#include <model.h>
#include <tokenizers_cpp.h>

std::mt19937 rng;
std::chrono::duration<double> raw_exec_seconds;

std::tuple<uint16_t *, uint16_t *>
get_cos_sin (int context_size, int pos_dim, const double theta,
    const std::string &rope_type, double partial_rotary_factor, double rope_scaling_factor)
{

  const double exponent = 1.0 / static_cast<double> (pos_dim);

  const float scale = 3.051804378628731e-05f;
  const int offset = -32768;

  std::vector<double> inv_freq (pos_dim);
  for (int j = 0; j < pos_dim; j++) {
    inv_freq[j] = 1.0 / std::pow (theta, j * exponent);
  }

  double attention_factor = 1.0;

  if (rope_type == "default") {
    // already computed
  } else if (rope_type == "linear" || rope_type == "proportional") {
    if (rope_scaling_factor > 1.0) {
      for (int j = 0; j < pos_dim; j++)
        inv_freq[j] /= rope_scaling_factor;
    }
  } else {
    // layter implementation for dynamic / yarn / longrope / llama3
  }

  uint16_t *cos_val =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  uint16_t *sin_val =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);

  for (int i = 0; i < context_size; i++) {
    for (int j = 0; j < pos_dim; j++) {
      const double freq = i * inv_freq[j];
      cos_val[i * pos_dim + j] =
          (std::cos(freq) * attention_factor) / scale - offset;
      sin_val[i * pos_dim + j] =
          (std::sin(freq) * attention_factor) / scale - offset;
    }
  }
  return std::make_tuple(cos_val, sin_val);
}

void process_key(uint8_t *pointer, int row, int column, uint8_t *dest, int idx,
                 int dest_row_length, int src_row_length) {
  // format: 1:1:col:row
  for (int i = 0; i < column; i++) {
    std::memcpy(dest + i * dest_row_length + idx, pointer + i * src_row_length,
                row);
  }
}

void process_value(uint8_t *pointer, int row, int column, uint8_t *dest,
                   int idx) {
  // format: 1:1:row:col
  for (int i = 0; i < row; i++) {
    std::memcpy(dest + (idx + i) * column, pointer + i * column, column);
  }
}

void fill_attention_mask_with_length(int rows, int columns, int length,
                                     uint16_t *attention_mask) {

  int index = 0;
  // TODO use std::fill_n?
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (i >= length) {
        attention_mask[index] = std::numeric_limits<uint16_t>::min();
      } else if (j >= columns - rows && j <= columns - rows + i) {
        attention_mask[index] = std::numeric_limits<uint16_t>::max();
      } else {
        attention_mask[index] = std::numeric_limits<uint16_t>::min();
      }
      index++;
    }
  }
}

void fill_attention_mask_with_prev_length(int rows, int columns, int length, 
                                          uint16_t *attention_mask) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < length; j++) {
      attention_mask[i * columns + j] = std::numeric_limits<uint16_t>::max();
    }
  }                                            
}

uint16_t *get_zero_memory(int size, int zero_point) {
  uint16_t *memory = (uint16_t *)allocate(size * sizeof(uint16_t));
  for (int i = 0; i < size; i++)
    memory[i] = zero_point;
  return memory;
}

int sample(uint16_t *pointer, int length, int *tokens, int number_of_tokens,
           float logit_scale, int logit_offset, float repetition_penalty,
           float temperature, float top_p, int top_k) {
  // Priority queue!
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                      std::greater<std::pair<int, int>>>
      top_k_elements;
  for (int i = 0; i < top_k && i < length; i++) {
    top_k_elements.push(std::make_pair(pointer[i], i));
  }
  for (int i = top_k; i < length; i++) {
    if (top_k_elements.top().first < pointer[i]) {
      top_k_elements.pop();
      top_k_elements.push(std::make_pair(pointer[i], i));
    }
  }
  length = top_k_elements.size();

  // Convert to float
  std::vector<int> indices(length);
  std::vector<float> logits(length);
  for (int i = 0; i < length; i++) {
    auto element = top_k_elements.top();
    logits[i] = (1.0 * element.first + logit_offset) * logit_scale;
    indices[i] = element.second;
    top_k_elements.pop();
  }

  for(unsigned int i=0;i<number_of_tokens;++i){
    const int t=tokens[i];
    for(int j=0;j<length;++j){
      if (indices[j] == tokens[i]){
	if(logits[j] >0.0f) logits[j] /=repetition_penalty;
	else logits[j] *= repetition_penalty;
	break;
      }
    }
  }


  std::vector<std::pair<int, float>> top_indices_and_logits (length);
  for (int i = 0; i < length; ++i) {
    if (temperature > 1e-5)
      logits[i] = logits[i] / temperature;
    top_indices_and_logits[i] = { i, logits[i] };
  }
  sort (top_indices_and_logits.begin (), top_indices_and_logits.end (),
      [] (auto &a, auto &b) { return a.second > b.second; });

  const float max_logit = top_indices_and_logits[0].second;
  std::vector<float> probs (length);
  float sum_exp = 0.0f;
  for (int i = 0; i < length; ++i) {
    probs[i] = std::exp (top_indices_and_logits[i].second - max_logit);
    sum_exp += probs[i];
  }
  if (sum_exp <= 0.0f)
    sum_exp = 1.0f;
  for (int i = 0; i < length; ++i) {
    probs[i] /= sum_exp;
  }

  float cum_prob = 0.0f;
  unsigned int top_index = 0;
  while (top_index < (unsigned)length && cum_prob < top_p) {
    cum_prob += probs[top_index];
    ++top_index;
  }
  if (top_index == 0)
    top_index = 1;

  // Apply Top-P: nuke beyond top_index
  for (int i = 0; i < length; ++i)
    logits[i] = -INFINITY;
  for (unsigned int i = 0; i < top_index; ++i) {
    logits[top_indices_and_logits[i].first] = top_indices_and_logits[i].second;
  }

  // Final softmax for sampling
  const float final_max = top_indices_and_logits[0].second;
  float final_sum_exp = 0.0f;
  for (int i = 0; i < length; ++i) {
    float ex = std::exp (logits[i] - final_max);
    final_sum_exp += ex;
    logits[i] = ex;
  }
  if (final_sum_exp <= 0.0f)
    final_sum_exp = 1.0f;
  for (int i = 0; i < length; ++i)
    logits[i] /= final_sum_exp;

  // Sample
  std::discrete_distribution<int> dist (logits.data (), logits.data () + length);
  return indices[dist (rng)];
}
