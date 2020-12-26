#include "bitmap_helpers.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/gen_op_registration.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <string>
#include <unistd.h>

int main(int argc, char *argv[]){
  const std::vector<std::string> args(argv + 1, argv + argc);
  float out[100];
  int inputDim[4];
  int outputDim[4];
  std::string model_path = args[0];
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  
  if (!model) {
    printf("Failed to mmap mdoel\n");
    exit(0);
  }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
  interpreter->SetNumThreads(1);

  for (int i = 0; i < 4; i++) {
    inputDim[i] = 1;
    outputDim[i] = 1;
  }
  inputDim[1] = 3;
  inputDim[2] = 32;
  inputDim[3] = 32;
  outputDim[3]=100;

  printf("input %d %d %d %d\n", inputDim[0], inputDim[1], inputDim[2],
         inputDim[3]);
  printf("output %d %d %d %d\n", outputDim[0], outputDim[1], outputDim[2],
         outputDim[3]);

  int output_number_of_pixels = 1;

  int wanted_channels = inputDim[0];
  int wanted_height = inputDim[1];
  int wanted_width = inputDim[2];

  for (int k = 0; k < 4; k++)
    output_number_of_pixels *= inputDim[k];

  int input = interpreter->inputs()[0];

  std::string img = args[1];
  
  uint8_t *in;
  float *output;
  in = tflite::label_image::read_bmp(img, &wanted_width, &wanted_height,
				     &wanted_channels);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tnesors!" << std::endl;
    return -2;
  }

  for (int l = 0; l < output_number_of_pixels; l++) {
    (interpreter->typed_tensor<float>(input))[l] =
      ((float)in[l] - 127.5f) / 127.5f;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cout << "Failed to invoke!" << std::endl;
    return -3;
  }
  
  output = interpreter->typed_output_tensor<float>(0);
  
  std::copy(output, output + 100, out);
  
  delete[] in;
  
  sleep(1);

  return 0;

}
