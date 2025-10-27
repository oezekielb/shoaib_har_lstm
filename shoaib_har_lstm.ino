#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "imu_sensor.h"
#include "shoaib_har_lstm.h"

// Tensor arena size — tune down/up as needed (start 48k)
constexpr int kTensorArenaSize = 17 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

// ---- Make resolver and interpreter static/globals so they outlive setup() ----
static tflite::MicroMutableOpResolver<22> microOpResolver;
static tflite::MicroInterpreter* tflInterpreter = nullptr;
static TfLiteTensor* tflInputTensor = nullptr;
static TfLiteTensor* tflOutputTensor = nullptr;
static const tflite::Model* tflModel = nullptr;

// Quantization parameters
float scale; // input quantization scale
int zero_point; // input quantization zero point

uint signal_size = 0;
void set_input_tensor(uint idx, float value) {
  int32_t q_value = (int32_t)round(value / scale) + zero_point;
  if (q_value < -128) q_value = -128;
  if (q_value > 127) q_value = 127;
  tflInputTensor->data.int8[idx] = (int8_t)q_value;
}

void handle_signal() {
  float ax, ay, az, gx, gy, gz;
  if (imuSensor.readAcceleration(ax, ay, az) && imuSensor.readGyroscope(gx, gy, gz)) {
    uint idx = signal_size * 6;
    // Fill input tensor with quantized values
    set_input_tensor(idx + 0, ax);
    set_input_tensor(idx + 1, ay);
    set_input_tensor(idx + 2, az);
    set_input_tensor(idx + 3, gx);
    set_input_tensor(idx + 4, gy);
    set_input_tensor(idx + 5, gz);
    signal_size++;
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Start");

  // Load model from generated header
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch");
    while (1);
  }

  // Register ops — add everything your model needs (example set)
  microOpResolver.AddShape();
  microOpResolver.AddPack();
  microOpResolver.AddTranspose();
  microOpResolver.AddFill();
  microOpResolver.AddDequantize();
  microOpResolver.AddWhile();
  microOpResolver.AddQuantize();
  microOpResolver.AddStridedSlice();
  microOpResolver.AddFullyConnected();
  microOpResolver.AddSoftmax();
  microOpResolver.AddLess();
  microOpResolver.AddLogicalAnd();
  microOpResolver.AddGather();
  microOpResolver.AddAdd();
  microOpResolver.AddSplit();
  microOpResolver.AddSub();
  microOpResolver.AddSlice();
  microOpResolver.AddExpandDims();
  microOpResolver.AddConcatenation();
  microOpResolver.AddLogistic();
  microOpResolver.AddMul();
  microOpResolver.AddTanh();
  // --- add other ops your model requires ---

  // Create the interpreter statically
  static tflite::MicroInterpreter static_interpreter(tflModel, microOpResolver, tensor_arena, kTensorArenaSize);
  tflInterpreter = &static_interpreter;

  Serial.println("Allocating tensors...");
  TfLiteStatus allocate_status = tflInterpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }
  Serial.println("Tensors allocated");

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  scale = tflInputTensor->params.scale;
  zero_point = tflInputTensor->params.zero_point;

  Serial.print("Input type: ");
  Serial.println(tflInputTensor->type);  // 9 for int8
  Serial.print("Input dims: ");
  for (int i=0;i<tflInputTensor->dims->size;i++) {
    Serial.print(tflInputTensor->dims->data[i]);
    Serial.print(i < tflInputTensor->dims->size-1 ? "x":"\n");
  }

  Serial.println("Initializing IMU...");
  imuSensor.debug(Serial);
  imuSensor.onInterrupt(handle_signal);
  // initialize the IMU
  if (!imuSensor.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(imuSensor.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(imuSensor.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println("Setup complete.");
}

void loop() {
  if (signal_size < 128) {
    // Serial.println("Collecting signal...");
    delay(100); // wait for more data
    return;
  }

  Serial.println("Running inference...");
  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Inference failed!");
    while(1);
  }

  // read outputs (if int8)
  int out0 = tflOutputTensor->dims->data[0];
  int out1 = tflOutputTensor->dims->data[1];
  Serial.println("Output values:");
  for (int i = 0; i < out1; i++) {
    Serial.print("Class ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(tflOutputTensor->data.int8[i]);
  }

  signal_size = 0; // reset for next signal
  delay(1000);
}
